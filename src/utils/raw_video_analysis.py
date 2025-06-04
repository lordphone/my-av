import os
import cv2
import ffmpeg
import json
import numpy as np
import subprocess
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from src.data.comma2k19dataset import Comma2k19Dataset

def get_video_metadata_ffprobe(video_path):
    """Get detailed video metadata using ffprobe with multiple approaches."""
    try:
        # First try to get frame count and duration using show_streams
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'format=duration:stream=nb_frames,r_frame_rate,codec_name',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        
        if result.returncode != 0:
            return None, f"FFprobe error: {result.stderr.strip()}"
        
        # Parse the JSON output
        try:
            data = json.loads(result.stdout)
            
            # Get format duration first (more reliable)
            format_duration = float(data.get('format', {}).get('duration', 0))
            
            # Get stream info
            if not data.get('streams') or len(data['streams']) == 0:
                return None, "No video stream found"
                
            stream = data['streams'][0]
            
            # Get frame count
            frame_count = int(stream.get('nb_frames', 0))
            
            # Get frame rate
            frame_rate = 0
            if 'r_frame_rate' in stream:
                try:
                    num, den = map(float, stream['r_frame_rate'].split('/'))
                    if den != 0:
                        frame_rate = num / den
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Calculate duration from frames and frame rate if needed
            if frame_count > 0 and frame_rate > 0 and format_duration <= 0:
                format_duration = frame_count / frame_rate
            
            # Calculate frame count from duration if needed
            if frame_count <= 0 and frame_rate > 0 and format_duration > 0:
                frame_count = int(format_duration * frame_rate)
            
            # If we still don't have frame count, try counting packets
            if frame_count <= 0:
                cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-count_frames',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_read_frames',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path
                ]
                count_result = subprocess.run(cmd, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, 
                                           text=True)
                if count_result.returncode == 0 and count_result.stdout.strip().isdigit():
                    frame_count = int(count_result.stdout.strip())
            
            return {
                'frames': frame_count,
                'duration': format_duration,
                'fps': frame_rate,
                'codec': stream.get('codec_name', 'unknown')
            }, None
            
        except (json.JSONDecodeError, KeyError) as e:
            return None, f"Failed to parse ffprobe output: {str(e)}"
        
    except Exception as e:
        return None, f"Error running ffprobe: {str(e)}"

def get_video_metadata(video_path):
    """Extract video metadata using multiple methods for reliability."""
    try:
        # First try to get detailed metadata using our custom ffprobe command
        metadata, error = get_video_metadata_ffprobe(video_path)
        
        # If that fails, fall back to python-ffmpeg
        if metadata is None:
            try:
                probe = ffmpeg.probe(video_path)
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                
                if not video_stream:
                    return None, "No video stream found"
                    
                duration = float(video_stream.get('duration', 0))
                codec_name = video_stream.get('codec_name', 'unknown')
                
                # Get frame rate
                frame_rate = 0
                if 'r_frame_rate' in video_stream:
                    try:
                        num, den = map(float, video_stream['r_frame_rate'].split('/'))
                        if den != 0:
                            frame_rate = num / den
                    except (ValueError, ZeroDivisionError):
                        pass
                
                # Try to get frame count
                frame_count = int(video_stream.get('nb_frames', 0))
                
                # If we have duration and frame rate but no frame count, calculate it
                if frame_count <= 0 and frame_rate > 0 and duration > 0:
                    frame_count = int(duration * frame_rate)
                
                metadata = {
                    'frames': frame_count,
                    'duration': duration,
                    'fps': frame_rate,
                    'codec': codec_name
                }
                
            except ffmpeg.Error as e:
                return None, f"FFprobe error: {e.stderr.decode() if e.stderr else str(e)}"
            except Exception as e:
                return None, f"Error reading video metadata: {str(e)}"
        
        # If we still don't have a frame count, try OpenCV as last resort
        if metadata['frames'] <= 0:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0:
                        metadata['frames'] = frame_count
                        
                        # If we didn't have duration, try to get it from OpenCV
                        if metadata['duration'] <= 0 and metadata['fps'] > 0:
                            metadata['duration'] = frame_count / metadata['fps']
                    
                    cap.release()
            except Exception as e:
                if 'cap' in locals():
                    cap.release()
                # Don't fail, just continue with what we have
                pass
        
        # Add the video path to the metadata
        metadata['path'] = video_path
        
        return metadata, None
        
    except Exception as e:
        return None, f"Error processing {video_path}: {str(e)}"

def analyze_videos(data_path, top_n=10, max_videos=None):
    """Analyze videos in the dataset and return statistics."""
    dataset = Comma2k19Dataset(data_path)
    total_segments = len(dataset)
    print(f"Found {total_segments} video segments to analyze")
    
    if max_videos and max_videos > 0:
        print(f"Processing first {min(max_videos, total_segments)} videos (use --max-videos 0 to process all)")
        total_segments = min(max_videos, total_segments)
    
    video_stats = []
    errors = []
    processed_paths = set()
    
    # Process each video segment
    video_files = []
    with tqdm(total=total_segments, desc="Analyzing videos") as pbar:
        for i, segment in enumerate(dataset):
            if max_videos and i >= max_videos:
                break
                
            video_path = segment.get("video_path")
            if not video_path:
                errors.append("Empty video path in segment")
                pbar.update(1)
                continue
                
            if not os.path.exists(video_path):
                errors.append(f"Video file not found: {video_path}")
                pbar.update(1)
                continue
                
            video_files.append(video_path)
            pbar.update(1)
    
    # Process each video file
    video_metadata = []
    for video_path in tqdm(video_files, desc="Analyzing videos"):
        metadata, error = get_video_metadata(video_path)
        if error:
            print(f"\nError processing {video_path}: {error}")
            errors.append(error)
            continue
        video_metadata.append(metadata)
    
    # Sort by frame count
    video_metadata.sort(key=lambda x: x['frames'])
    
    # Print summary
    print("\n" + "="*80)
    print(f"Analysis complete. Processed {len(video_metadata)} unique videos with {len(errors)} errors.\n")
    
    if not video_metadata:
        print("No valid videos found for analysis.")
        return []
    
    # Print top N videos with fewest frames
    print(f"Top {min(top_n, len(video_metadata))} videos with fewest frames:")
    print("-"*120)
    print(f"{'Frames':<10} {'Duration (s)':<12} {'FPS':<10} {'Codec':<10} {'Path'}")
    print("-"*120)
    
    for meta in video_metadata[:top_n]:
        # Extract the relevant part of the path for display
        rel_path = os.path.relpath(meta['path'], data_path)
        print(f"{meta['frames']:<10} {meta['duration']:<12.2f} {meta['fps']:<10.2f} {meta['codec']:<10} {rel_path}")
    
    # Print summary statistics
    frames = [m['frames'] for m in video_metadata if m['frames'] > 0]
    if frames:
        total_frames = sum(frames)
        avg_frames = total_frames / len(frames)
        min_frames = min(frames)
        max_frames = max(frames)
        
        print("\n" + "="*120)
        print("SUMMARY STATISTICS (only including videos with valid frame counts):")
        print(f"Total videos processed: {len(video_metadata)}")
        print(f"Videos with valid frame counts: {len(frames)}")
        print(f"Total frames across all videos: {total_frames:,}")
        print(f"Average frames per video: {avg_frames:,.0f}")
        print(f"Minimum frames: {min_frames} ({min_frames/20/60:.1f} minutes at 20fps)")
        print(f"Maximum frames: {max_frames} ({max_frames/20/60:.1f} minutes at 20fps)")
        
        # Print frame count distribution
        print("\nFrame count distribution:")
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        for p in percentiles:
            value = np.percentile(frames, p)
            minutes = value / 20 / 60  # Convert frames to minutes at 20fps
            print(f"{p:3}%: {value:,.0f} frames ({minutes:.1f} min at 20fps)")
    
    # Print error summary if there were errors
    if errors:
        error_summary = {}
        for error in errors:
            error_summary[error] = error_summary.get(error, 0) + 1
            
        # Sort by count descending
        sorted_errors = sorted(error_summary.items(), key=lambda x: x[1], reverse=True)
        
        print("\nError Summary:")
        for error, count in sorted_errors[:10]:  # Show top 10 most common errors
            print(f"- {count}× {error}")
        if len(sorted_errors) > 10:
            print(f"... and {len(sorted_errors) - 10} more unique error types.")
    
    return video_metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video segments in the dataset.')
    parser.add_argument('--data-path', type=str, default="/home/lordphone/my-av/data/raw/comma2k19/",
                       help='Path to the dataset directory')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of videos with fewest frames to display')
    parser.add_argument('--max-videos', type=int, default=100,
                       help='Maximum number of videos to process (0 for all)')
    
    args = parser.parse_args()
    
    print(f"Starting video analysis on: {args.data_path}")
    try:
        analyze_videos(
            data_path=args.data_path,
            top_n=args.top_n,
            max_videos=args.max_videos if args.max_videos > 0 else None
        )
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user. Generating report with current data...")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise