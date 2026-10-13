import numpy as np

from myav.data.frames import EON_H, EON_W, get_warp_matrix, i420_frame_size, warp_yuv_frame


def test_warp_shape_and_crop():
    yuv = np.random.randint(0, 256, i420_frame_size(EON_W, EON_H), dtype=np.uint8)
    out = warp_yuv_frame(yuv, get_warp_matrix())
    y = yuv[:EON_W * EON_H].reshape(EON_H, EON_W)
    assert out.shape == (6, 128, 256)
    assert out.dtype == np.uint8
    # with zero calibration and equal focal lengths the warp is a pure crop
    x0, y0 = round(EON_W / 2 - 256), round(EON_H / 2 - 47.6)
    np.testing.assert_array_equal(out[0], y[y0:y0 + 256:2, x0:x0 + 512:2])
