import numpy as np

def empty(shape, dtype=np.float32):
    return np.empty(shape, dtype=dtype)

def im2col(img, k_h, k_w, s_h, s_w):
    """Transform padded image into column matrix.
    :param img: padded inputs of shape (B, in_h, in_w, in_c)
    :param k_h: kernel height
    :param k_w: kernel width
    :param s_h: stride height
    :param s_w: stride width
    :return col: column matrix of shape (B*out_h*out_w, k_h*k_w*inc)
    """
    batch_sz, h, w, in_c = img.shape
    # calculate result feature map size
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    # allocate space for column matrix
    col = empty((batch_sz * out_h * out_w, k_h * k_w * in_c))
    # fill in the column matrix
    batch_span = out_w * out_h
    for r in range(out_h):
        r_start = r * s_h
        matrix_r = r * out_w
        for c in range(out_w):
            c_start = c * s_w
            patch = img[:, r_start: r_start+k_h, c_start: c_start+k_w, :]
            patch = patch.reshape(batch_sz, -1)
            col[matrix_r+c::batch_span, :] = patch
    return col


def get_padding_2d(in_shape, k_shape, mode):

    def get_padding_1d(w, k):
        if mode == "SAME":
            pads = (w - 1) + k - w
            half = pads // 2
            padding = (half, half) if pads % 2 == 0 else (half, half + 1)
        else:
            padding = (0, 0)
        return padding

    h_pad = get_padding_1d(in_shape[0], k_shape[0])
    w_pad = get_padding_1d(in_shape[1], k_shape[1])
    return (0, 0), h_pad, w_pad, (0, 0)