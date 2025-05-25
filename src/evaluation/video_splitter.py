import os
import math
import moviepy
from moviepy import *


def split_video_into_chunks(video_path, chunk_duration=5):
    video_filename, video_ext = os.path.splitext(os.path.basename(video_path))
    input_video_directory = os.path.dirname(os.path.abspath(video_path))
    output_folder_path = os.path.join(input_video_directory, "3_seconds")

    os.makedirs(output_folder_path, exist_ok=True)

    clip = VideoFileClip(video_path)
    total_duration = clip.duration

    num_chunks = math.ceil(total_duration / chunk_duration)

    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)

        if total_duration - start_time < 0.1 and i > 0:
            break

        subclip = clip.subclipped(start_time, end_time)
        chunk_filename = f"{video_filename}_{i+1}"
        output_chunk_path = os.path.join(output_folder_path, f"{chunk_filename}.mp4")
        subclip.write_videofile(output_chunk_path, logger=None, codec="libx264")

    clip.close()
    print(f"\nFinished processing {video_filename}.")


if __name__ == "__main__":
    print(moviepy.__version__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Configuration ---
    input_videos_parent_folder_name = "box_of_lies" 
    chunk_length_seconds = 3
    # ---------------------

    input_dir_path = os.path.join(script_dir, input_videos_parent_folder_name)
    
    if not os.path.isdir(input_dir_path):
        print(f"Error: Input directory not found: {input_dir_path}")
        print(f"Please create it or check the 'input_videos_parent_folder_name' variable.")
        exit()

    for filename in os.listdir(input_dir_path):
        if filename.lower().endswith('.mp4'):
            video_file_path = os.path.join(input_dir_path, filename)
            print(f"\n--- Processing video: {filename} ---")
            split_video_into_chunks(video_file_path, chunk_length_seconds)