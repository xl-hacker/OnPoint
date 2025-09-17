import argparse
import cv2
import models
import numpy as np
from pathlib import Path
import torch
import utils

def predict_video_segment(model, video_path, start_time, end_time, preprocess, device):
    """
    Predict class for a video segment
    """
    # Extract frames
    frames = utils.extract_frames_uniform(
        video_path, num_frames=16, start_time=start_time, end_time=end_time)
    
    # Prepare input
    input_tensor = utils.prepare_3d_input(frames, preprocess)
    input_tensor = input_tensor.to(device, non_blocking=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def create_overlapping_windows(video_duration, window_duration=3.0, overlap=1.0):
    windows = []
    start_time = 0.0
    
    while start_time < video_duration:
        end_time = min(start_time + window_duration, video_duration)
        windows.append((start_time, end_time))
        start_time += (window_duration - overlap)
    
    return windows

def get_video_info(video_path):
    """
    Get video duration and FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return duration, fps, total_frames

def create_visualization_video(input_video_path, predictions, output_video_path, fps=30):
    """
    Create visualization video with predictions overlaid
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if actual_fps > 0:
        fps = actual_fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    current_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find prediction for current time
        current_prediction = None
        for (start_time, end_time), (pred_class, confidence) in predictions:
            if start_time <= current_time <= end_time:
                current_prediction = (pred_class, confidence)
                break
        
        # Overlay prediction on frame
        if current_prediction:
            pred_class, confidence = current_prediction
            
            # Create overlay text with meaningful labels
            class_label = "Background" if pred_class == 0 else "Point ongoing"
            class_text = f"Status: {class_label}"
            conf_text = f"Confidence: {confidence:.3f}"
            
            # Color based on class (green for class 0, red for class 1)
            color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)
            
            # Position overlay at bottom of frame for better visibility
            overlay_x = 20
            overlay_y = height - 100  # 100 pixels from bottom
            text_width = 280
            text_height = 70
            
            # Add background rectangle with semi-transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + text_width, overlay_y + text_height), (0, 0, 0), -1)
            cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + text_width, overlay_y + text_height), color, 3)
            
            # Blend overlay with original frame (semi-transparent)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Add text
            cv2.putText(frame, class_text, (overlay_x + 10, overlay_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, conf_text, (overlay_x + 10, overlay_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add time indicator at top right
            time_text = f"Time: {current_time:.1f}s"
            cv2.putText(frame, time_text, (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        frame_count += 1
        current_time = frame_count / fps
    
    cap.release()
    out.release()
    print(f"Visualization video saved to: {output_video_path}")
    print(f"Wrote {frame_count} frames at {fps:.1f} FPS")

def create_class1_compilation_video(input_video_path, predictions, output_video_path, fps=30, buffer_seconds=1.0):
    """
    Create a compilation video with only class 1 clips and buffer time
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Filter class 1 predictions
    class1_predictions = [(start_time, end_time) for (start_time, end_time), (pred_class, _) in predictions if pred_class == 1]
    
    if not class1_predictions:
        print("No class 1 predictions found. Creating empty compilation video.")
        cap.release()
        out.release()
        return
    
    print(f"Creating compilation with {len(class1_predictions)} class 1 clips...")
    
    # Sort predictions by start time
    class1_predictions.sort(key=lambda x: x[0])
    
    # Merge consecutive clips and add buffers
    merged_clips = []
    current_start, current_end = class1_predictions[0]
    
    for start_time, end_time in class1_predictions[1:]:
        # If this clip is consecutive (or very close), merge it
        if start_time - current_end <= buffer_seconds:
            current_end = end_time
        else:
            # Add buffer to the end of current clip and start of next clip
            merged_clips.append((current_start, current_end + buffer_seconds))
            current_start = max(0, start_time - buffer_seconds)
            current_end = end_time
    
    # Add the last clip
    merged_clips.append((current_start, current_end + buffer_seconds))
    
    print(f"Merged into {len(merged_clips)} clips with buffers")
    
    # Extract and write frames for each clip
    total_frames_written = 0
    
    for i, (start_time, end_time) in enumerate(merged_clips):
        print(f"Processing clip {i+1}/{len(merged_clips)}: {start_time:.1f}s - {end_time:.1f}s")
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames (no overlay text)
        frames_to_write = end_frame - start_frame
        for frame_idx in range(frames_to_write):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write frame without any overlay
            out.write(frame)
            total_frames_written += 1
    
    cap.release()
    out.release()
    
    compilation_duration = total_frames_written / fps
    print(f"Class 1 compilation video saved to: {output_video_path}")
    print(f"Compilation duration: {compilation_duration:.2f}s ({total_frames_written} frames)")

def analyze_video(
    video_path, model_path, output_path=None, compilation_path=None, window_duration=3.0, overlap=1.0):
    """
    Analyze video using overlapping windows and create visualization
    """
    print(f"Analyzing video: {video_path}")
    
    device = utils.get_device()
    model, preprocess = models.load_trained_model(model_path, device=device)
    
    duration, fps, total_frames = get_video_info(video_path)
    print(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}, Total frames: {total_frames}")
    
    # Create overlapping windows
    windows = create_overlapping_windows(duration, window_duration, overlap)
    print(f"Created {len(windows)} overlapping windows")
    
    # Analyze each window
    predictions = []
    print("\nAnalyzing video segments...")
    
    for i, (start_time, end_time) in enumerate(windows):
        print(f"Window {i+1}/{len(windows)}: {start_time:.1f}s - {end_time:.1f}s")
        frames = utils.extract_frames_uniform(
            video_path, num_frames=16, start_time=start_time, end_time=end_time)
        
        input_tensor = utils.prepare_3d_input(frames, preprocess)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        predictions.append(((start_time, end_time), (pred_class, confidence)))
        print(f"  Prediction: Class {pred_class}, Confidence: {confidence:.3f}")
    
    if output_path is None:
        input_path = Path(video_path)
        output_path = input_path.parent / f"{input_path.stem}_analyzed.mp4"
    
    if compilation_path is None:
        input_path = Path(video_path)
        compilation_path = input_path.parent / f"{input_path.stem}_class1_compilation.mp4"
    
    # Create visualization video
    print(f"\nCreating visualization video...")
    create_visualization_video(video_path, predictions, output_path, fps)
    
    # Create class 1 compilation video
    print(f"\nCreating class 1 compilation video...")
    create_class1_compilation_video(video_path, predictions, compilation_path, fps, buffer_seconds=1.0)
    
    print(f"\n=== Analysis Summary ===")
    class_0_count = sum(1 for _, (pred_class, _) in predictions if pred_class == 0)
    class_1_count = sum(1 for _, (pred_class, _) in predictions if pred_class == 1)
    
    print(f"Total windows analyzed: {len(predictions)}")
    print(f"Class 0 predictions: {class_0_count}")
    print(f"Class 1 predictions: {class_1_count}")
    print(f"Average confidence: {np.mean([conf for _, (_, conf) in predictions]):.3f}")

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Analyze video using overlapping windows")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--model", default="point_classifier.pth", help="Path to trained model")
    parser.add_argument("--output", help="Path to output overlay visualization video")
    parser.add_argument("--compilation", help="Path to output class 1 compilation video")
    parser.add_argument("--window-duration", type=float, default=3.0, help="Duration of each window in seconds")
    parser.add_argument("--overlap", type=float, default=1.0, help="Overlap between windows in seconds")
    
    args = parser.parse_args()
    
    analyze_video(
        args.video_path,
        args.model,
        args.output,
        args.compilation,
        args.window_duration,
        args.overlap,
    )

if __name__ == "__main__":
    main()
