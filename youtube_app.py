import yt_dlp #type : ignore
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

download_folder = filedialog.askdirectory(title="Select Download Folder")
if not download_folder:
    print("No folder selected. Exiting.")
    exit()

print("Selected download quality :")
print("1. Best Quality")
print("2. audio only")
print("3. custom format (e.g. 1080p or 720p)")

quality_choice = input("Enter your choice (1/2/3): ")

ydl_opts = {
    'outtmpl': f'{download_folder}/%(title)s.%(ext)s',
}

if quality_choice == '1':
    ydl_opts['format'] = 'best'
elif quality_choice == '2':
    ydl_opts['format'] = 'bestaudio'
elif quality_choice == '3':
    custom_format = input("Enter custom format (e.g., 1080p, 720p): ")
    ydl_opts['format'] = f'bestvideo[height<={custom_format}]+bestaudio/best[height<={custom_format}]'
else:
    print("Invalid choice. Exiting.")
    ydl_opts['format'] = 'best'

url = input("Enter the video URL: ")

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print(f"Download completed!{download_folder}!")