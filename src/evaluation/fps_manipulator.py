import os
from moviepy import VideoFileClip

# This file was used to even out FPS to a specific value, during testing

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "test_low_fps")
new_fps = 10
video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv")

def process_videos(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Create temporary filename
                temp_path = os.path.join(folder_path, f"TEMP_{filename}")
                
                # Process video
                video = VideoFileClip(file_path)
                
                # Write to temp file (preserves audio codec)
                video.write_videofile(
                    temp_path,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,  # Disable progress logs
                    fps=10
                )
                
                # Overwrite original file
                os.replace(temp_path, file_path)
                print(f"Successfully processed: {filename}")
                
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")
                # Clean up temporary file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    process_videos(folder_path)