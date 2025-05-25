import cv2
import os
import random

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return frames

def process_videos(video_paths, target_image_count: bool|int = False, output_dir="output_frames"):
    all_frames = []
    frame_number = 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from each video
    for video_path in video_paths:
        frames = extract_frames(video_path)
        print(f"Extracted {len(frames)} from {video_path}")
        all_frames.extend(frames)

    # Shuffle all frames
    random.shuffle(all_frames)
    
    # Reduce amount of frames before saving to JPG
    if target_image_count:
        frame_number = len(all_frames)
        all_frames = all_frames[:target_image_count]
        print(f"Frame count reduced from {frame_number} to {len(all_frames)}")

    # Save each frame as JPG
    for idx, frame in enumerate(all_frames):
        digits = len(str(target_image_count))
        filename = os.path.join(output_dir, f"frame_{idx+1:0{digits}d}.jpg")
        cv2.imwrite(filename, frame)

    print(f"Saved {len(all_frames)} frames to '{output_dir}'")

def get_file_names(directory: str) -> list[str]:
    file_extensions = (".mp4")
    video_file_names = []

    # Make a list of files ending in ".mp4"
    for file in os.listdir(directory):
        if file.endswith(file_extensions):
            video_file_names.append(os.path.join(directory, file))
        else:
            continue
    return video_file_names

# Example usage
if __name__ == "__main__":
    video_files = get_file_names("video")
    process_videos(video_files, target_image_count=500)