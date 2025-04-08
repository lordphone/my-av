# tool to read frames from the comma2k19 driving video. Videos are in HEVC format, 20 frames per second.


import ffmpeg
import numpy as np
import os
import subprocess

class FrameReader:
    """
    Simple frame reader class to read frames from a video file.
    """
    def __init__(self, video_path):
        self.video_path = video_path

        # Check that video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
        probe = ffmpeg.probe(video_path)
        
        # Gather video metadata
        self.width = probe['streams'][0]['width']
        self.height = probe['streams'][0]['height']
        self.fps = eval(probe['streams'][0]['r_frame_rate'])

        # Initialize ffmpeg process
        self.ffmpeg_process = None

    def get(self, frame_idx, pix_fmt='rgb24'):
        """
        Get a frame from the video file, specified by the frame index.

        Args:
            frame_idx (int): The index of the frame to retrieve.
            pix_fmt (str): The pixel format to use. Default is 'rgb24'.

        Returns:
            np.ndarray: The requested frame as a numpy array.
        """
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.kill()

        # Start ffmpeg process
        out, _ = (
            ffmpeg
            .input(self.video_path, ss=frame_idx / self.fps)
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt,  vframes=1)
            .run(capture_stdout=True, quiet=True)
        )

        if not out:
            raise ValueError(f"Failed to extract frame {frame_idx} from video {self.video_path}")

        channels = 3 if pix_fmt == 'rgb24' else 1
        frame = (
            np.frombuffer(out, np.uint8)
            .reshape([self.height, self.width, channels])
        )
        return frame