# op_frames.py
# Converts comma2k19 road camera video into openpilot's driving model input format.
# Mirrors openpilot/common/transformations/model.py and tinygrad examples/openpilot/compile_warp.py

import subprocess
import numpy as np
import cv2

# openpilot MEDMODEL input frame
MODEL_W, MODEL_H = 512, 256
MODEL_FL = 910.0
MODEL_CY = 47.6

# comma2k19 was recorded on an eon, same road camera as openpilot's _neo_config
EON_W, EON_H, EON_FL = 1164, 874, 910.0

VIEW_FROM_DEVICE = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])

# scales a full res warp matrix to the half res U/V planes (same as compile_warp.py)
UV_SCALE = np.array([[1.0, 1.0, 0.5],
                     [1.0, 1.0, 0.5],
                     [2.0, 2.0, 1.0]])


def intrinsics(fl, cx, cy):
    return np.array([[fl, 0., cx],
                     [0., fl, cy],
                     [0., 0., 1.]])


MODEL_K = intrinsics(MODEL_FL, 0.5 * MODEL_W, MODEL_CY)
EON_K = intrinsics(EON_FL, 0.5 * EON_W, 0.5 * EON_H)


def rot_from_euler(rpy):
    """Same convention as openpilot's rot_from_euler (roll, pitch, yaw)."""
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def get_warp_matrix(device_from_calib_euler=(0., 0., 0.), cam_k=EON_K):
    """Maps model frame pixels -> camera pixels, like openpilot's get_warp_matrix."""
    calib_from_model = np.linalg.inv(MODEL_K @ VIEW_FROM_DEVICE)
    camera_from_calib = cam_k @ VIEW_FROM_DEVICE @ rot_from_euler(device_from_calib_euler)
    return (camera_from_calib @ calib_from_model).astype(np.float32)


def _warp(plane, M, size):
    # nearest neighbour + clamped borders, matching the tinygrad warp on device
    return cv2.warpPerspective(plane, M, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def i420_frame_size(width, height):
    cw, ch = (width + 1) // 2, (height + 1) // 2
    return width * height + 2 * cw * ch


def warp_yuv_frame(yuv, M, width=EON_W, height=EON_H):
    """
    yuv: flat I420 frame from ffmpeg yuv420p (chroma planes are rounded up for odd sizes)
    returns: openpilot packed frame [6, 128, 256] uint8
    """
    cw, ch = (width + 1) // 2, (height + 1) // 2
    y = yuv[:width * height].reshape(height, width)
    u = yuv[width * height:width * height + cw * ch].reshape(ch, cw)
    v = yuv[width * height + cw * ch:width * height + 2 * cw * ch].reshape(ch, cw)

    M_uv = (M * UV_SCALE).astype(np.float32)
    y = _warp(y, M, (MODEL_W, MODEL_H))
    u = _warp(u, M_uv, (MODEL_W // 2, MODEL_H // 2))
    v = _warp(v, M_uv, (MODEL_W // 2, MODEL_H // 2))

    # same channel packing as frames_to_tensor in compile_warp.py
    return np.stack([y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2], u, v])


def read_warped_frames(video_path, M=None, width=EON_W, height=EON_H):
    """Decodes a whole video and returns warped frames [N, 6, 128, 256] uint8."""
    M = get_warp_matrix() if M is None else M
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-vsync", "0", "-i", video_path,
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        stdout=subprocess.PIPE)
    frame_size = i420_frame_size(width, height)
    frames = []
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        frames.append(warp_yuv_frame(np.frombuffer(buf, dtype=np.uint8), M, width, height))
    proc.wait()
    return np.stack(frames) if frames else np.zeros((0, 6, MODEL_H // 2, MODEL_W // 2), np.uint8)
