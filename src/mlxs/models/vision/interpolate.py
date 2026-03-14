"""Bicubic interpolation for vision position embedding resizing.

Pure MLX implementation — no custom kernels needed.
"""

from __future__ import annotations

import mlx.core as mx


def bicubic_interpolate(
    x: mx.array,
    size: tuple[int, int],
) -> mx.array:
    """Resize a 4D tensor (N, C, H, W) using bicubic interpolation.

    Simple implementation using bilinear as MLX doesn't have native bicubic.
    For position embedding interpolation, bilinear is a good approximation.
    """
    n, c, h_in, w_in = x.shape
    h_out, w_out = size

    if h_in == h_out and w_in == w_out:
        return x

    # Transpose to (N, H, W, C) for processing
    x = x.transpose(0, 2, 3, 1)

    # Create output grid coordinates
    h_scale = h_in / h_out
    w_scale = w_in / w_out

    # Source coordinates
    h_coords = (mx.arange(h_out, dtype=mx.float32) + 0.5) * h_scale - 0.5
    w_coords = (mx.arange(w_out, dtype=mx.float32) + 0.5) * w_scale - 0.5

    h_coords = mx.clip(h_coords, 0, h_in - 1)
    w_coords = mx.clip(w_coords, 0, w_in - 1)

    # Bilinear interpolation
    h0 = mx.floor(h_coords).astype(mx.int32)
    w0 = mx.floor(w_coords).astype(mx.int32)
    h1 = mx.minimum(h0 + 1, h_in - 1)
    w1 = mx.minimum(w0 + 1, w_in - 1)

    hf = (h_coords - h0.astype(mx.float32))[:, None]  # (h_out, 1)
    wf = (w_coords - w0.astype(mx.float32))[None, :]  # (1, w_out)

    # Gather corners for all batch items
    results = []
    for i in range(n):
        img = x[i]  # (H, W, C)
        top_left = img[h0][:, w0]        # (h_out, w_out, C)
        top_right = img[h0][:, w1]
        bottom_left = img[h1][:, w0]
        bottom_right = img[h1][:, w1]

        top = top_left * (1 - wf[..., None]) + top_right * wf[..., None]
        bottom = bottom_left * (1 - wf[..., None]) + bottom_right * wf[..., None]
        result = top * (1 - hf[..., None]) + bottom * hf[..., None]
        results.append(result)

    output = mx.stack(results, axis=0)  # (N, h_out, w_out, C)
    return output.transpose(0, 3, 1, 2)  # (N, C, h_out, w_out)
