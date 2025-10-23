"""
Unified Brand Detection + RAG Chat Dashboard with AWS S3 Upload
Run with: streamlit run unified_dashboard.py
"""

import streamlit as st
import os
import tempfile
import cv2
from ultralytics import YOLO
import pandas as pd
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# -------------------------
# Configuration
# -------------------------
# YOLO Model Path - USING PRE-TRAINED MODEL
MODEL_PATH = "best_model.pt"  # This will download automatically

# PostgreSQL Configuration
DB_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'database': os.getenv('PG_DB', 'brand_detection'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASS', 'vino2003'),
    'port': int(os.getenv('PG_PORT', 5432))
}

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Brand Detection + RAG Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# S3 Upload Function
# -------------------------
def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload a file to an S3 bucket"""
    try:
        s3_client = boto3.client(
            's3',                                      
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except ClientError as e:
        st.error(f"❌ S3 Upload failed: {e}")
        return None

# -------------------------
# Database Functions - FIXED VERSION
# -------------------------
def get_connection():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

def create_table():
    """Create detections table in PostgreSQL"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cur = conn.cursor()
        
        # Check if table exists first
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'brand_detections'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            cur.execute("""
                CREATE TABLE brand_detections (
                    id SERIAL PRIMARY KEY,
                    video_name VARCHAR(255),
                    frame INTEGER,
                    timestamp_s REAL,
                    detected_logo_name VARCHAR(100),
                    confidence REAL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    frame_width INTEGER,
                    frame_height INTEGER,
                    detection_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX idx_video_name ON brand_detections(video_name)
            """)
            
            cur.execute("""
                CREATE INDEX idx_detected_logo ON brand_detections(detected_logo_name)
            """)
            
            conn.commit()
            st.success("✅ Database table created successfully!")
        else:
            st.info("ℹ️ Database table already exists")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return False

def save_detections_to_db(video_name, detections_list):
    """Save all detections to PostgreSQL database"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cur = conn.cursor()
        
        data = [
            (
                video_name,
                d['Frame'],
                d['Timestamp (s)'],
                d['Detected_Logo_Name'],
                d['Confidence'],
                d.get('bbox_x1', 0),
                d.get('bbox_y1', 0),
                d.get('bbox_x2', 0),
                d.get('bbox_y2', 0),
                d.get('frame_width', 0),
                d.get('frame_height', 0)
            )
            for d in detections_list
        ]
        
        query = """
            INSERT INTO brand_detections 
            (video_name, frame, timestamp_s, detected_logo_name, confidence,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_width, frame_height)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cur, query, data)
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Failed to save to database: {str(e)}")
        return False

def get_all_data():
    """Get ALL data from table"""
    try:
        conn = get_connection()
        if not conn:
            return [], []
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM brand_detections
            ORDER BY detection_datetime DESC
        """)
        
        results = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        
        return [dict(zip(colnames, row)) for row in results], colnames
    
    except Exception as e:
        st.error(f"❌ Error getting data: {e}")
        return [], []

def get_table_info():
    """Get column names and data types"""
    try:
        conn = get_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'brand_detections'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        cursor.close()
        conn.close()
        return columns
    except Exception as e:
        st.error(f"❌ Error getting table info: {e}")
        return []

def get_database_stats():
    """Get database statistics"""
    try:
        conn = get_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        stats = {}
        
        # Total rows
        cursor.execute("SELECT COUNT(*) FROM brand_detections")
        stats['total_rows'] = cursor.fetchone()[0]
        
        # Unique videos
        cursor.execute("SELECT COUNT(DISTINCT video_name) FROM brand_detections")
        stats['unique_videos'] = cursor.fetchone()[0]
        
        # Unique brands
        cursor.execute("SELECT COUNT(DISTINCT detected_logo_name) FROM brand_detections")
        stats['unique_brands'] = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM brand_detections")
        avg_conf = cursor.fetchone()[0]
        stats['avg_confidence'] = round(avg_conf, 2) if avg_conf else 0
        
        cursor.close()
        conn.close()
        return stats
    except Exception as e:
        return {}

# -------------------------
# RAG Functions
# -------------------------
def format_context_summary(columns, all_data):
    """Format database info into a SUMMARIZED context for AI"""
    context = f"Brand Detection Database Summary\n"
    context += f"Total Rows: {len(all_data)}\n"
    context += f"Total Columns: {len(columns)}\n\n"
    
    context += "Columns:\n"
    for col_name, col_type in columns:
        context += f"  - {col_name} ({col_type})\n"
    
    # Create summary statistics instead of raw data
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Brand summary
        if 'detected_logo_name' in df.columns:
            brand_summary = df['detected_logo_name'].value_counts().head(10)
            context += f"\nTop Brands:\n"
            for brand, count in brand_summary.items():
                context += f"  - {brand}: {count} detections\n"
        
        # Video summary
        if 'video_name' in df.columns:
            video_summary = df['video_name'].value_counts().head(5)
            context += f"\nVideos:\n"
            for video, count in video_summary.items():
                context += f"  - {video}: {count} detections\n"
        
        # Confidence summary
        if 'confidence' in df.columns:
            context += f"\nConfidence Statistics:\n"
            context += f"  - Average: {df['confidence'].mean():.2f}\n"
            context += f"  - Range: {df['confidence'].min():.2f} to {df['confidence'].max():.2f}\n"
    
    else:
        context += "\nNo data available in database."
    
    return context

def ask_groq(question, context):
    """Call Groq AI for RAG with error handling"""
    if not GROQ_API_KEY:
        return """❌ GROQ_API_KEY not found! Add it to your .env file."""
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = """You are a brand detection analysis assistant. Answer questions based ONLY on the provided database context."""
        
        user_prompt = f"""Database Context:
{context}

Question: {question}

Answer:"""
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# -------------------------
# YOLO Model Loading
# -------------------------
@st.cache_resource
def load_model():
    try:
        st.info("🔄 Loading YOLO model (this may take a moment)...")
        model = YOLO(MODEL_PATH)
        st.success("✅ YOLO model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# -------------------------
# Header
# -------------------------
st.markdown('<p class="main-header">🎥 Brand Detection + AI Chat System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload Videos • Detect Brands • Store in Database • Ask AI Questions</p>', unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Database info
    st.markdown("### 🗄️ Database")
    st.info(f"""
    **Host:** {DB_CONFIG['host']}:{DB_CONFIG['port']}  
    **Database:** {DB_CONFIG['database']}  
    **Table:** brand_detections
    """)
    
    # Test connection
    if st.button("🔌 Test Database", use_container_width=True):
        with st.spinner("Testing..."):
            conn = get_connection()
            if conn:
                st.success("✅ Connected!")
                conn.close()
            else:
                st.error("❌ Failed!")
    
    # Initialize DB
    if st.button("🔧 Initialize Database", use_container_width=True):
        if create_table():
            st.success("✅ Database ready!")
    
    st.divider()
    
    # AI Status
    st.markdown("### 🤖 AI Assistant")
    if GROQ_API_KEY:
        st.success("✅ Groq AI Ready")
    else:
        st.error("❌ API Key Missing")
    
    st.divider()
    
    # S3 Status
    st.markdown("### ☁️ AWS S3")
    if S3_BUCKET_NAME and AWS_ACCESS_KEY_ID:
        st.success(f"✅ S3 Ready: {S3_BUCKET_NAME}")
    else:
        st.warning("⚠️ S3 not configured")
    
    st.divider()
    
    # Actions
    st.markdown("### 🔄 Actions")
    if st.button("♻️ Refresh All Data", use_container_width=True):
        st.session_state.db_loaded = False
        st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - What brands were detected?
    - Which video has most detections?
    - Show brands with confidence > 0.8
    - What's the average confidence?
    - Compare all detected brands
    """)

# -------------------------
# Initialize Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pause" not in st.session_state:
    st.session_state.pause = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
if "all_data" not in st.session_state:
    st.session_state.all_data = []
if "columns" not in st.session_state:
    st.session_state.columns = []

# -------------------------
# Load Database Data
# -------------------------
def load_database_data():
    """Load database data into session state"""
    try:
        # First ensure table exists
        create_table()
        
        # Then load data
        all_data, colnames = get_all_data()
        columns = get_table_info()
        
        if all_data is not None:
            st.session_state.all_data = all_data
            st.session_state.columns = columns
            st.session_state.db_loaded = True
            return True
        return False
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return False

# Load database on startup
if not st.session_state.db_loaded:
    with st.spinner("📡 Checking database..."):
        load_database_data()

# -------------------------
# Main Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Video Processing",
    "💬 AI Chat Assistant", 
    "📊 Analytics Dashboard",
    "🔍 Database Explorer"
])

# -------------------------
# TAB 1: Video Processing - FIXED S3 UPLOAD
# -------------------------
with tab1:
    st.header("📹 Brand Detection from Video")
    
    # Load model
    model = load_model()
    if not model:
        st.error("❌ Failed to load YOLO model. Please check your model path.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("📤 Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        video_name = uploaded_file.name
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        # 🔄 FIX: Upload to S3 BEFORE processing (when file still exists)
        s3_url = None
        if S3_BUCKET_NAME and AWS_ACCESS_KEY_ID:
            s3_key = f"uploads/{video_name}"
            with st.spinner(f"☁️ Uploading {video_name} to S3..."):
                s3_url = upload_to_s3(video_path, S3_BUCKET_NAME, s3_key)
                if s3_url:
                    st.success(f"✅ Uploaded to S3: [View]({s3_url})")
                else:
                    st.warning("⚠️ S3 upload failed, continuing with processing...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Failed to open video")
            os.unlink(video_path)
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"📹 **{video_name}** | Frames: {total_frames} | FPS: {fps:.2f} | Resolution: {frame_width}x{frame_height}")
        
        stframe = st.empty()
        progress_bar = st.progress(0, text="⏳ Ready to process...")
        
        frame_count = 0
        detections_list = []
        
        # Video processing loop - process every 5th frame for speed
        while cap.isOpened():
            if st.session_state.pause:
                if st.session_state.last_frame is not None:
                    stframe.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
                time.sleep(0.1)
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to speed up processing
            if frame_count % 5 == 0:
                # YOLO detection
                results = model(frame)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.session_state.last_frame = annotated_frame_rgb
                
                # Collect detections
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = model.names[cls_id]
                            timestamp = round(frame_count / fps, 2)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detections_list.append({
                                "Frame": frame_count,
                                "Timestamp (s)": timestamp,
                                "Detected_Logo_Name": label,
                                "Confidence": round(conf, 2),
                                "bbox_x1": int(x1),
                                "bbox_y1": int(y1),
                                "bbox_x2": int(x2),
                                "bbox_y2": int(y2),
                                "frame_width": frame_width,
                                "frame_height": frame_height
                            })
                
                # Display frame
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update progress
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress, text=f"Processing: {frame_count}/{total_frames}")
        
        cap.release()
        
        # 🔄 FIX: Only delete the temp file AFTER we're completely done with it
        os.unlink(video_path)
        
        # Save results
        st.divider()
        
        if detections_list:
            df = pd.DataFrame(detections_list)
            st.success(f"✅ Video processing completed! Found {len(detections_list)} detections.")
            
            # Save to database
            with st.spinner("💾 Saving to database..."):
                if save_detections_to_db(video_name, detections_list):
                    st.success("✅ Detections saved to database!")
                    
                    # 🔄 RELOAD DATABASE DATA
                    with st.spinner("🔄 Updating dashboard..."):
                        load_database_data()
                    
                else:
                    st.error("❌ Failed to save to database")
            
            # Display results
            st.subheader("📊 Detection Results")
            st.dataframe(df[['Frame', 'Timestamp (s)', 'Detected_Logo_Name', 'Confidence']], 
                        use_container_width=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Detections", len(df))
            with col2:
                st.metric("Unique Objects", df['Detected_Logo_Name'].nunique())
            with col3:
                st.metric("Frames Processed", frame_count)
            with col4:
                st.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")
            
            # Brand summary
            st.subheader("📈 Detection Summary")
            brand_summary = df.groupby('Detected_Logo_Name').agg({
                'Confidence': ['count', 'mean'],
                'Timestamp (s)': ['min', 'max']
            }).round(2)
            brand_summary.columns = ['Count', 'Avg Conf', 'First (s)', 'Last (s)']
            brand_summary['Duration (s)'] = brand_summary['Last (s)'] - brand_summary['First (s)']
            st.dataframe(brand_summary, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"{video_name}_detections.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.warning("⚠️ No objects detected in the video")
    else:
        st.info("👆 Upload a video to start detection")

# -------------------------
# TAB 2: AI Chat Assistant
# -------------------------
with tab2:
    st.header("💬 AI-Powered Database Q&A")
    
    # Check database state
    if not st.session_state.db_loaded:
        with st.spinner("🔄 Loading database..."):
            load_database_data()
    
    if st.session_state.db_loaded:
        stats = get_database_stats()
        
        st.success(f"✅ Database loaded: **{stats.get('total_rows', 0)}** rows, **{stats.get('unique_brands', 0)}** brands")
        
        # Display database summary
        with st.expander("📋 View Database Summary"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", stats.get('total_rows', 0))
            with col2:
                st.metric("Videos", stats.get('unique_videos', 0))
            with col3:
                st.metric("Brands", stats.get('unique_brands', 0))
            with col4:
                st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about detected objects..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing..."):
                    context = format_context_summary(st.session_state.columns, st.session_state.all_data)
                    response = ask_groq(prompt, context)
                    st.markdown(response)
            
            # Add response
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("⚠️ Database not loaded. Please process a video first.")

# -------------------------
# TAB 3: Analytics Dashboard
# -------------------------
with tab3:
    st.header("📊 Analytics Dashboard")
    
    # Check database state
    if not st.session_state.db_loaded:
        with st.spinner("🔄 Loading database..."):
            load_database_data()
    
    if st.session_state.db_loaded and st.session_state.all_data:
        stats = get_database_stats()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Rows", stats.get('total_rows', 0))
        with col2:
            st.metric("🎥 Videos", stats.get('unique_videos', 0))
        with col3:
            st.metric("🏷️ Brands", stats.get('unique_brands', 0))
        with col4:
            st.metric("⭐ Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
        
        st.divider()
        
        # Convert to dataframe
        df = pd.DataFrame(st.session_state.all_data)
        
        # Brand distribution
        if 'detected_logo_name' in df.columns and len(df) > 0:
            st.subheader("🏷️ Object Distribution")
            brand_counts = df['detected_logo_name'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(
                    x=brand_counts.index, y=brand_counts.values,
                    labels={'x': 'Object', 'y': 'Count'},
                    title='Detection Frequency',
                    color=brand_counts.values,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                if len(brand_counts) > 0:
                    fig_pie = px.pie(
                        values=brand_counts.values,
                        names=brand_counts.index,
                        title='Detection Distribution'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence distribution
        if 'confidence' in df.columns and len(df) > 0:
            st.subheader("⭐ Confidence Distribution")
            fig_hist = px.histogram(
                df, x='confidence', nbins=20,
                title='Confidence Scores',
                labels={'confidence': 'Confidence', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
    else:
        st.warning("📭 No data available. Process a video first!")

# -------------------------
# TAB 4: Database Explorer
# -------------------------
with tab4:
    st.header("🔍 Database Explorer")
    
    # Check database state
    if not st.session_state.db_loaded:
        with st.spinner("🔄 Loading database..."):
            load_database_data()
    
    if st.session_state.db_loaded and st.session_state.all_data:
        df = pd.DataFrame(st.session_state.all_data)
        
        st.info(f"📊 Showing {len(df)} rows from database")
        st.dataframe(df, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Data",
            csv,
            f"brand_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.warning("📭 No data to display. Process a video first!")

# -------------------------
# Footer
# -------------------------
st.divider()
st.caption("🚀 Powered by YOLOv8 + Groq AI + PostgreSQL | Made By: VINOTHKUMAR S")