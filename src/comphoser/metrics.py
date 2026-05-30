"""Image fidelity metrics used by ComPhoser evaluation."""

from __future__ import annotations

import math
import threading
from typing import Any, Mapping

import numpy as np
from PIL import Image


# LPIPS model cache. Loaded lazily on first compute_lpips call, reused across all
# subsequent calls. Keyed on (net, device_str) so periodic validation and the
# standalone evaluator share a single in-process model. The lock guards the
# first-load race in case validation fires concurrently.
_LPIPS_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_LPIPS_LOAD_LOCK = threading.Lock()


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Return an RGB float32 array in the 0..255 range."""

    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32)


def validate_same_image_shape(output_image: Image.Image, target_image: Image.Image, *, context: str) -> None:
    output_shape = image_to_rgb_array(output_image).shape
    target_shape = image_to_rgb_array(target_image).shape
    if output_shape != target_shape:
        raise ValueError(
            f"{context} requires generated output and ground truth to share the same array shape, "
            f"but got {output_shape} vs {target_shape}."
        )


def compute_psnr_db(output_image: Image.Image, target_image: Image.Image) -> float:
    output_array, target_array = _paired_rgb_arrays(output_image, target_image, metric_name="PSNR")
    # Difference in float64 to avoid float32 round-off in the squared error (R36).
    diff = output_array.astype(np.float64) - target_array.astype(np.float64)
    mse = float(np.mean(np.square(diff), dtype=np.float64))
    if mse == 0.0:
        return 100.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


# SSIM Gaussian local-statistics window, matched to scikit-image
# `structural_similarity(..., gaussian_weights=True, sigma=1.5)`:
# scipy's gaussian_filter uses radius = int(truncate*sigma + 0.5) = 5, i.e. an
# 11x11 window. scikit-image discards a `radius`-pixel border from the SSIM map
# before averaging (`crop`); we do the same so numbers are comparable to the
# de-facto reference implementation (R03).
_SSIM_GAUSSIAN_SIGMA = 1.5
_SSIM_GAUSSIAN_TRUNCATE = 3.5
_SSIM_GAUSSIAN_RADIUS = int(_SSIM_GAUSSIAN_TRUNCATE * _SSIM_GAUSSIAN_SIGMA + 0.5)


def compute_ssim(output_image: Image.Image, target_image: Image.Image) -> float:
    """Mean RGB SSIM over an 11x11 Gaussian window, scikit-image-comparable.

    Uses sigma=1.5 / truncate=3.5 Gaussian local statistics (an 11x11 window) and
    crops the `radius`-pixel border from the SSIM map before averaging, matching
    `skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5)`."""

    output_array, target_array = _paired_rgb_arrays(output_image, target_image, metric_name="SSIM")
    x = output_array.astype(np.float64)
    y = target_array.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    radius = _SSIM_GAUSSIAN_RADIUS
    channel_scores = []
    for channel in range(3):
        x_channel = x[..., channel]
        y_channel = y[..., channel]
        mu_x = _gaussian_filter_2d(x_channel)
        mu_y = _gaussian_filter_2d(y_channel)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = _gaussian_filter_2d(x_channel * x_channel) - mu_x_sq
        sigma_y_sq = _gaussian_filter_2d(y_channel * y_channel) - mu_y_sq
        sigma_xy = _gaussian_filter_2d(x_channel * y_channel) - mu_xy

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / denominator
        # Discard the Gaussian border (skimage `crop`) when the image is large enough.
        if ssim_map.shape[0] > 2 * radius and ssim_map.shape[1] > 2 * radius:
            ssim_map = ssim_map[radius:-radius, radius:-radius]
        channel_scores.append(float(np.mean(ssim_map, dtype=np.float64)))
    return float(np.mean(channel_scores, dtype=np.float64))


def compute_delta_e_2000(output_image: Image.Image, target_image: Image.Image) -> float:
    output_array, target_array = _paired_rgb_arrays(output_image, target_image, metric_name="Delta E 2000")
    output_lab = rgb_to_lab(output_array)
    target_lab = rgb_to_lab(target_array)
    return float(np.mean(ciede2000(output_lab, target_lab), dtype=np.float64))


_LPIPS_MIN_SPATIAL_DIM = 64


def compute_lpips(
    output_image: Image.Image,
    target_image: Image.Image,
    *,
    net: str = "alex",
    device: str | None = None,
) -> float:
    """LPIPS perceptual distance (lower = more similar). Uses the standalone `lpips`
    package with the AlexNet backbone by default. Model weights are lazily loaded
    and cached per (net, device).

    Inputs smaller than 64 px on either spatial dimension are bilinearly upsampled
    before being fed to the network (AlexNet's five pool layers would otherwise
    collapse the feature map to zero spatial extent)."""

    import torch
    import torch.nn.functional as F

    output_array, target_array = _paired_rgb_arrays(output_image, target_image, metric_name="LPIPS")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_lpips_model(net=net, device=device)

    def _to_tensor(arr: np.ndarray) -> Any:
        # LPIPS expects shape (B, 3, H, W) with values in [-1, 1].
        scaled = (arr.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(scaled).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    output_tensor = _to_tensor(output_array)
    target_tensor = _to_tensor(target_array)

    if min(output_tensor.shape[-2:]) < _LPIPS_MIN_SPATIAL_DIM:
        size = _LPIPS_MIN_SPATIAL_DIM
        output_tensor = F.interpolate(output_tensor, size=(size, size), mode="bilinear", align_corners=False)
        target_tensor = F.interpolate(target_tensor, size=(size, size), mode="bilinear", align_corners=False)

    with torch.no_grad():
        score = model(output_tensor, target_tensor)
    return float(score.item())


def _get_lpips_model(*, net: str, device: str) -> Any:
    key = (net, device)
    if key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[key]
    with _LPIPS_LOAD_LOCK:
        if key in _LPIPS_MODEL_CACHE:
            return _LPIPS_MODEL_CACHE[key]
        import lpips

        model = lpips.LPIPS(net=net, verbose=False).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _LPIPS_MODEL_CACHE[key] = model
        return model


def compute_image_metrics(output_image: Image.Image, target_image: Image.Image) -> dict[str, float]:
    return {
        "psnr_db": compute_psnr_db(output_image, target_image),
        "ssim": compute_ssim(output_image, target_image),
        "delta_e_2000": compute_delta_e_2000(output_image, target_image),
        "lpips_alex": compute_lpips(output_image, target_image),
    }


def unavailable_image_metrics() -> dict[str, None]:
    return {
        "psnr_db": None,
        "ssim": None,
        "delta_e_2000": None,
        "lpips_alex": None,
    }


def image_metric_units() -> Mapping[str, str]:
    return {
        "psnr_db": "dB",
        "ssim": "unitless",
        "delta_e_2000": "delta_e",
        "lpips_alex": "unitless",
    }


def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_array, dtype=np.float64) / 255.0
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    xyz = linear_rgb @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    ).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    f_xyz = np.where(xyz > 0.008856, np.cbrt(xyz), (7.787 * xyz) + (16.0 / 116.0))
    lab = np.empty_like(f_xyz)
    lab[..., 0] = (116.0 * f_xyz[..., 1]) - 16.0
    lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return lab


def ciede2000(lab_1: np.ndarray, lab_2: np.ndarray) -> np.ndarray:
    l1, a1, b1 = np.moveaxis(np.asarray(lab_1, dtype=np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(lab_2, dtype=np.float64), -1, 0)

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = np.power(c_bar, 7.0)
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + (25.0**7))))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)
    c_bar_prime = (c1_prime + c2_prime) / 2.0

    h1_prime = _hue_angle_degrees(b1, a1_prime)
    h2_prime = _hue_angle_degrees(b2, a2_prime)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    h_diff = h2_prime - h1_prime
    h_diff = np.where(h_diff > 180.0, h_diff - 360.0, h_diff)
    h_diff = np.where(h_diff < -180.0, h_diff + 360.0, h_diff)
    delta_h_prime = np.where(
        (c1_prime * c2_prime) == 0.0,
        0.0,
        2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(np.deg2rad(h_diff / 2.0)),
    )

    l_bar_prime = (l1 + l2) / 2.0
    h_sum = h1_prime + h2_prime
    h_bar_prime = np.where(
        (c1_prime * c2_prime) == 0.0,
        h_sum,
        np.where(
            np.abs(h1_prime - h2_prime) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )

    t = (
        1.0
        - (0.17 * np.cos(np.deg2rad(h_bar_prime - 30.0)))
        + (0.24 * np.cos(np.deg2rad(2.0 * h_bar_prime)))
        + (0.32 * np.cos(np.deg2rad((3.0 * h_bar_prime) + 6.0)))
        - (0.20 * np.cos(np.deg2rad((4.0 * h_bar_prime) - 63.0)))
    )
    delta_theta = 30.0 * np.exp(-np.square((h_bar_prime - 275.0) / 25.0))
    c_bar_prime7 = np.power(c_bar_prime, 7.0)
    r_c = 2.0 * np.sqrt(c_bar_prime7 / (c_bar_prime7 + (25.0**7)))
    s_l = 1.0 + ((0.015 * np.square(l_bar_prime - 50.0)) / np.sqrt(20.0 + np.square(l_bar_prime - 50.0)))
    s_c = 1.0 + (0.045 * c_bar_prime)
    s_h = 1.0 + (0.015 * c_bar_prime * t)
    r_t = -np.sin(np.deg2rad(2.0 * delta_theta)) * r_c

    return np.sqrt(
        np.square(delta_l_prime / s_l)
        + np.square(delta_c_prime / s_c)
        + np.square(delta_h_prime / s_h)
        + (r_t * (delta_c_prime / s_c) * (delta_h_prime / s_h))
    )


def _paired_rgb_arrays(
    output_image: Image.Image,
    target_image: Image.Image,
    *,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    output_array = image_to_rgb_array(output_image)
    target_array = image_to_rgb_array(target_image)
    if output_array.shape != target_array.shape:
        raise ValueError(
            f"Cannot compute {metric_name} for images with different array shapes: "
            f"{output_array.shape} vs {target_array.shape}"
        )
    return output_array, target_array


def _gaussian_filter_2d(values: np.ndarray) -> np.ndarray:
    # scipy is required for SSIM. Previously a bare `except Exception: return values`
    # silently turned SSIM into ungated local statistics (~0.69/~1.0 garbage) on any
    # failure; narrow to ImportError and fail loudly so SSIM is never silently wrong (R03).
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover - exercised only without scipy installed
        raise ImportError(
            "compute_ssim requires scipy (scipy.ndimage.gaussian_filter). Install scipy to compute SSIM."
        ) from exc
    return gaussian_filter(values, sigma=_SSIM_GAUSSIAN_SIGMA, truncate=_SSIM_GAUSSIAN_TRUNCATE, mode="reflect")


def _hue_angle_degrees(b_values: np.ndarray, a_values: np.ndarray) -> np.ndarray:
    angle = np.rad2deg(np.arctan2(b_values, a_values))
    return np.where(angle < 0.0, angle + 360.0, angle)


__all__ = [
    "ciede2000",
    "compute_delta_e_2000",
    "compute_image_metrics",
    "compute_lpips",
    "compute_psnr_db",
    "compute_ssim",
    "image_metric_units",
    "image_to_rgb_array",
    "rgb_to_lab",
    "unavailable_image_metrics",
    "validate_same_image_shape",
]
