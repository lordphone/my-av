import ffmpeg
import numpy as np
import os
import subprocess

class FrameReader:
    """
    Optimized frame reader for comma2k19 driving videos.

    Compared to my first basic frame reader, improvements include:
    - Checking for CUDA availability and using it for hardware acceleration if possible.
    - Using ffmpeg to read frames in batches instead of one at a time.
    """
    def __init__(self, video_path, use_cuda=False, batch_size=400):
        self.video_path = video_path
        self.use_cuda = use_cuda and self._is_cuda_available()
        self.batch_frames = None
        self.batch_start_idx = -1
        self.batch_size = batch_size  # Number of frames to process in one batch
        
        # Check that video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
            
        # Get video metadata
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")
            
        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])
        self.fps = eval(video_stream['r_frame_rate'])
        
        # Get total number of frames
        self.num_frames = 0
        # Use ffmpeg to count the total number of frames
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
                    self.video_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            self.num_frames = int(result.stdout.strip())
        except subprocess.SubprocessError as e:
            print(f"FFmpeg stderr output: {e.stderr}")  # Log the stderr output for debugging
            raise RuntimeError(f"Failed to count frames in video {self.video_path}: {e}")

        # Initialize iteration state
        self._current_frame = 0
    
    def _is_cuda_available(self):
        """Check if CUDA is available for hardware acceleration"""
        try:
            # Try running a simple ffmpeg command with CUDA
            subprocess.run(
                ["ffmpeg", "-hwaccel", "cuda", "-hwaccel_device", "0", "-f", "lavfi", 
                 "-i", "nullsrc", "-t", "1", "-f", "null", "-"],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _load_batch(self, start_idx):
        """Load a batch of frames starting from start_idx."""
        batch_size = min(self.batch_size, self.num_frames - start_idx)
        if batch_size <= 0:
            return None

        # Set up ffmpeg command without 'ss' for HEVC compatibility
        input_args = {}
        if self.use_cuda:
            input_args.update({'hwaccel': 'cuda', 'hwaccel_device': '0'})

        # Use frame-based seeking instead of timestamp-based seeking
        try:
            out, err = (
                ffmpeg
                .input(self.video_path, **input_args)
                .filter('select', f'gte(n,{start_idx})')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=batch_size, vsync='0')
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error while loading batch starting at frame {start_idx}: {e.stderr.decode('utf-8')}")
            raise

        if not out:
            return None

        # Reshape into individual frames
        frames = np.frombuffer(out, np.uint8).reshape([batch_size, self.height, self.width, 3])
        return frames
    
    def get(self, frame_idx):
        """
        Get a frame from the video file using the frame index.

        Args:
            frame_idx (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: The requested frame as a numpy array.
        """
        if not 0 <= frame_idx < self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
        
        # Check if the requested frame is in the current batch
        if (self.batch_frames is not None and 
            self.batch_start_idx <= frame_idx < self.batch_start_idx + len(self.batch_frames)):
            return self.batch_frames[frame_idx - self.batch_start_idx]
        
        # If not, load a new batch centered around the requested frame
        batch_start = (frame_idx // self.batch_size) * self.batch_size
        self.batch_frames = self._load_batch(batch_start)
        self.batch_start_idx = batch_start
        
        if self.batch_frames is None:
            raise ValueError(f"Failed to extract frame {frame_idx} from video {self.video_path}")
            
        # Return the requested frame from the new batch
        return self.batch_frames[frame_idx - self.batch_start_idx]
    
    def get_frames(self, start_idx, num_frames):
        """
        Get multiple consecutive frames starting from start_idx.
        Much more efficient than calling get() repeatedly.

        Args:
            start_idx (int): Starting frame index
            num_frames (int): Number of frames to retrieve

        Returns:
            np.ndarray: Array of frames with shape [num_frames, height, width, 3]
        """
        if not 0 <= start_idx < self.num_frames:
            raise IndexError(f"Start index {start_idx} out of range [0, {self.num_frames-1}]")
            
        num_frames = min(num_frames, self.num_frames - start_idx)
        
        # If all frames can be loaded in a single batch, do that
        if num_frames <= self.batch_size:
            batch = self._load_batch(start_idx)
            if batch is None:
                raise ValueError(f"Failed to extract frames starting at {start_idx}")
            return batch[:num_frames]
        
        # Otherwise, load in multiple batches
        frames = np.zeros((num_frames, self.height, self.width, 3), dtype=np.uint8)
        for i in range(0, num_frames, self.batch_size):
            batch_size = min(self.batch_size, num_frames - i)
            batch = self._load_batch(start_idx + i)
            if batch is None:
                raise ValueError(f"Failed to extract frames starting at {start_idx + i}")
            frames[i:i+batch_size] = batch[:batch_size]
            
        return frames
    
    def __iter__(self):
        """Make the FrameReader iterable"""
        self._current_frame = 0
        return self
    
    def __next__(self):
        """Returns the next frame in the video"""
        if self._current_frame >= self.num_frames:
            raise StopIteration
            
        # For sequential access, we can optimize by prefetching batches
        if (self.batch_frames is None or 
            self._current_frame >= self.batch_start_idx + len(self.batch_frames) or 
            self._current_frame < self.batch_start_idx):
            
            # Get a new batch starting at the current frame index
            batch_start = (self._current_frame // self.batch_size) * self.batch_size
            self.batch_frames = self._load_batch(batch_start)
            self.batch_start_idx = batch_start
            
            if self.batch_frames is None:
                raise ValueError(f"Failed to extract frame {self._current_frame}")
        
        # Get the frame from the current batch
        frame = self.batch_frames[self._current_frame - self.batch_start_idx]
        self._current_frame += 1
        return frame
    
    def __len__(self):
        """Returns the total number of frames in the video"""
        return self.num_frames
    
    def reset(self):
        """Reset the iterator to the beginning of the video"""
        self._current_frame = 0