# fps_manipulator.py

To use this file configure the folder path relative to the scripts location. You may set target FPS via 'new_fps'.

# compare_faces.py

To use this file use two image_paths. Encoding is expensive, so only use this if you need each encoding only once. Threshhold controlls strictness of face matchings. 

# face_counter.py

This file goes through all faces which have to be previously extracted, and saves the matches between the person that was seen the most in a scene. As Face Recognition is not perfect, matches have to manually reviewed for precision.

# display_video.py

Since courtroom videos where chunked, this video loader makes manual review of scenes easier, by playing all chunks in order. Type the scene name during execution.
 
# extract_and_save_faces.py

This extracts all faces from a all videos in the specified subfolder of the deception_detection folder. We only keep the person we find the most in a scene.

# video_splitter.py

This script splits videos in a specified folder into evenly sized chunks which length you can control via 'chunk_length_seconds'.

# video_loader.py

This file loads single videos and displays them. 