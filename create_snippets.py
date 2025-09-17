import cv2
import os
import argparse
import json
import ast
from pathlib import Path

def read_timestamps_from_file(timestamp_file):
    """
    Read timestamp ranges from a file.
    Each line should contain a timestamp range in the format [start_time, end_time]

    This is the format of the labeled data for each video. Only annotate
    times where a point / rally is ongoing. Everything else is assumed to be background.
    
    Args:
        timestamp_file (str): Path to the file containing timestamp ranges
    
    Returns:
        list: List of [start_time, end_time] pairs
    """
    timestamps = []
    
    try:
        with open(timestamp_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        timestamp_range = ast.literal_eval(line)
                        if isinstance(timestamp_range, list) and len(timestamp_range) == 2:
                            start_time, end_time = timestamp_range
                            if start_time < end_time:
                                timestamps.append([start_time, end_time])
                            else:
                                print(f"Warning: Line {line_num}: start_time ({start_time}) >= end_time ({end_time})")
                        else:
                            print(f"Warning: Line {line_num}: Invalid format, expected [start_time, end_time]")
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Line {line_num}: Could not parse '{line}': {e}")
        
        print(f"Successfully read {len(timestamps)} timestamp ranges from {timestamp_file}")
        return timestamps
    
    except FileNotFoundError:
        print(f"Error: Timestamp file {timestamp_file} not found")
        return []
    except Exception as e:
        print(f"Error reading timestamp file: {e}")
        return []

def create_labels_from_timestamps(total_clips, clip_duration, label_timestamps):
    """
    Create labels for clips based on timestamp ranges.
    
    Args:
        total_clips (int): Total number of clips
        clip_duration (float): Duration of each clip in seconds
        label_timestamps (list): List of [start_time, end_time] pairs where label should be 1
    
    Returns:
        list: List of labels (0 or 1) for each clip
    """
    labels = [0] * total_clips  # Initialize all labels to 0
    
    for start_time, end_time in label_timestamps:
        # Calculate which clips overlap with this time range
        start_clip = int(start_time // clip_duration)
        end_clip = int(end_time // clip_duration)
        
        # Ensure we don't go beyond the number of clips
        start_clip = max(0, start_clip)
        end_clip = min(total_clips - 1, end_clip)
        
        # Set labels to 1 for all clips in this range
        for clip_idx in range(start_clip, end_clip + 1):
            labels[clip_idx] = 1
    
    return labels

def split_video_into_clips(video_path, output_folder, label_timestamps, clip_duration=3.0):
    """
    Split a video into clips of specified duration and create labels.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Folder to save the clips
        clip_duration (float): Duration of each clip in seconds (default: 3.0)
        label_timestamps (list): List of [start_time, end_time] pairs where label should be 1
    """
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    label_0_folder = os.path.join(output_folder, "0")
    label_1_folder = os.path.join(output_folder, "1")
    Path(label_0_folder).mkdir(parents=True, exist_ok=True)
    Path(label_1_folder).mkdir(parents=True, exist_ok=True)
    
    import time
    import hashlib
    
    video_path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    timestamp = int(time.time())
    unique_id = f"{timestamp}_{video_path_hash}"
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    frames_per_clip = int(fps * clip_duration)
    
    total_clips = total_frames // frames_per_clip
    
    print(f"\nSplitting into {total_clips} clips of {clip_duration} seconds each...")
    
    labels = create_labels_from_timestamps(total_clips, clip_duration, label_timestamps)
    print(f"Labels created: {labels}")
    
    video_name = Path(video_path).stem
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    clip_count = 0
    frame_count = 0
    
    temp_output_path = os.path.join(output_folder, f"{video_name}_{unique_id}_clip_{clip_count:06d}.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % frames_per_clip == 0:
            out.release()
            
            if labels and clip_count < len(labels):
                label = labels[clip_count]
                if label == 1:
                    final_output_path = os.path.join(label_1_folder, f"{video_name}_{unique_id}_clip_{clip_count:06d}.mp4")
                else:
                    final_output_path = os.path.join(label_0_folder, f"{video_name}_{unique_id}_clip_{clip_count:06d}.mp4")
                
                os.rename(temp_output_path, final_output_path)
                print(f"Created clip {clip_count:06d} (label {label}): {final_output_path}")
            else:
                # If no labels, put in label 0 folder
                final_output_path = os.path.join(label_0_folder, f"{video_name}_{unique_id}_clip_{clip_count:06d}.mp4")
                os.rename(temp_output_path, final_output_path)
                print(f"Created clip {clip_count:06d}: {final_output_path}")
            
            clip_count += 1
            
            # Start new clip if there are more frames
            if frame_count < total_frames:
                temp_output_path = os.path.join(output_folder, f"{video_name}_{unique_id}_clip_{clip_count:06d}.mp4")
                out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    out.release()
    cap.release()
    
    print(f"\nSuccessfully created {clip_count} clips in folder: {output_folder}")
    print(f"Each clip is approximately {clip_duration} seconds long")
    print(f"Clips organized in folders:")
    print(f"  Label 0: {label_0_folder}")
    print(f"  Label 1: {label_1_folder}")

def main():
    parser = argparse.ArgumentParser(description='Split a video into short clips with labels')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output', '-o', default='training_clips', 
                       help='Output folder for clips (default: training_clips)')
    parser.add_argument('--duration', '-d', type=float, default=3.0,
                       help='Duration of each clip in seconds (default: 3.0)')
    parser.add_argument('--timestamps', '-t', type=str,
                        help='Path to file containing timestamp ranges (one per line: [start_time, end_time])')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} does not exist")
        return
    
    label_timestamps = read_timestamps_from_file(args.timestamps)
    if not label_timestamps:
        print("Error: No valid timestamp ranges found in the provided file.")
        return
    
    labels = split_video_into_clips(
        args.video_path, args.output, label_timestamps, args.duration)
    
    if labels:
        print(f"\nFinal labels: {labels}")
        print(f"Number of clips with label 1: {sum(labels)}")
        print(f"Number of clips with label 0: {len(labels) - sum(labels)}")

if __name__ == "__main__":
    main()
