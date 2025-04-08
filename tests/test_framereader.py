import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.framereader import FrameReader

def test_framereader():
    video_path = "/home/lordphone/my-av/tests/video.mp4"
    
    try:
        # Initialize the FrameReader
        reader = FrameReader(video_path)
        print(f"Video metadata: width={reader.width}, height={reader.height}, fps={reader.fps}")

        # Read and print the shape of a few frames
        for frame_idx in [0, 10, 20]:
            frame = reader.get(frame_idx)
            print(f"Frame {frame_idx} shape: {frame.shape}")
            print(f"Frame {frame_idx} pixel data (first 10 pixels): {frame.flatten()[:10]}")

        print("FrameReader test passed.")
    except Exception as e:
        print(f"FrameReader test failed: {e}")

if __name__ == "__main__":
    test_framereader()