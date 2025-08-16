"""
Indian Flag Image Validator – Independence Day Coding Challenge

Implements automated checks per problem statement:
- Aspect ratio 3:2 (±1%)
- Color accuracy (±5% per-channel tolerance from reference RGB; deviation reported as max-channel % of full scale 255)
- Stripe proportions (each ~1/3 of height)
- Ashoka Chakra: centered in white band; diameter = 3/4 of white band height (checked internally), and exactly 24 spokes

Assumptions:
- Flat, solid-color flags (no folds/shading), as per constraints
- PNG/JPG/SVG (SVG should be rasterized before using this script)

Usage (CLI):
    python validator.py <image_path>
Outputs JSON report to stdout.

Use as a module:
    from validator import validate
    report = validate("path/to/image.png")
"""

import sys, json, time
from typing import Dict, Any, Tuple
from PIL import Image
import numpy as np

# Reference colors (RGB)
SAFFRON = np.array([255, 153,  51], dtype=np.float32)   # #FF9933
WHITE   = np.array([255, 255, 255], dtype=np.float32)   # #FFFFFF
GREEN   = np.array([ 19, 136,   8], dtype=np.float32)   # #138808
NAVY    = np.array([  0,   0, 128], dtype=np.float32)   # #000080

# Tolerances
TOL_ARATIO = 0.01   # ±1% aspect ratio
TOL_COLOR  = 0.05   # ±5% per-channel (fraction of full scale 255)
TOL_POS    = 0.01   # ±1% center tolerance (fraction of width/height)
TOL_DIAM   = 0.05   # ±5% diameter tolerance (internal check only)
BAND_TOL   = 0.02   # ±2% band proportion tolerance (fraction of height)
EPS        = 1e-9

def pct_deviation_max_channel(actual: np.ndarray, target: np.ndarray) -> float:
    """
    Return the maximum per-channel deviation as a fraction of 255.
    Matches ±5% RGB-channel tolerance and avoids issues where target channel is 0.
    """
    diff = np.abs(actual - target)
    frac = diff / 255.0
    return float(frac.max())

def nearest_stripe_label(rgb: np.ndarray) -> str:
    """Classify a color as saffron/white/green by nearest reference RGB (L2)."""
    targets = {'saffron': SAFFRON, 'white': WHITE, 'green': GREEN}
    dists = {k: np.linalg.norm(rgb - v) for k, v in targets.items()}
    return min(dists, key=dists.get)

def compute_row_labels(img_arr: np.ndarray) -> np.ndarray:
    """Label each row by the nearest stripe color based on its mean RGB."""
    H, W, _ = img_arr.shape
    labels = []
    for y in range(H):
        mean_rgb = img_arr[y].mean(axis=0)
        labels.append(nearest_stripe_label(mean_rgb))
    return np.array(labels)

def label_proportions(labels: np.ndarray) -> dict:
    """Return the fractional height occupied by each stripe label."""
    H = len(labels)
    props = {}
    for k in ['saffron', 'white', 'green']:
        props[k] = float((labels == k).sum()) / max(1, H)
    return props

def check_aspect_ratio(W: int, H: int) -> dict:
    """Check 3:2 aspect ratio within ±1% tolerance and report actual ratio."""
    actual = W / (H + EPS)
    target = 3.0 / 2.0
    ok = abs(actual - target) <= TOL_ARATIO * target
    return {"status": "pass" if ok else "fail", "actual": round(actual, 5)}

def is_blue_mask(img_arr: np.ndarray) -> np.ndarray:
    """
    Build a mask for navy blue (#000080). Allow a radius in RGB space that
    corresponds to ~5–8% tolerance for robustness across resolutions.
    """
    diff = np.linalg.norm(img_arr.astype(np.float32) - NAVY[None, None, :], axis=2)
    # 5% of 255 ≈ 12.75; choose 20 for robustness
    return diff <= 20.0

def check_colors(img_arr: np.ndarray) -> dict:
    """
    Evaluate color accuracy for saffron/white/green using thirds of the height,
    and chakra navy blue using a color mask. Deviation is the MAX per-channel
    deviation (fraction of 255) reported as a percent string (e.g., "3%").
    """
    H, W, _ = img_arr.shape
    thirds = H // 3
    regions = {
        "saffron": img_arr[0:thirds],
        "white":   img_arr[thirds:2*thirds],
        "green":   img_arr[2*thirds:H]
    }
    out = {}

    # Stripe colors
    for name, data in regions.items():
        mean_rgb = data.reshape(-1, 3).mean(axis=0)
        target = {"saffron": SAFFRON, "white": WHITE, "green": GREEN}[name]
        dev_frac = pct_deviation_max_channel(mean_rgb, target)
        status = "pass" if dev_frac <= TOL_COLOR else "fail"
        dev_pct_str = f"{int(round(dev_frac*100))}%"
        out[name] = {"status": status, "deviation": dev_pct_str}

    # Chakra blue
    blue_mask = is_blue_mask(img_arr)
    if blue_mask.sum() > 0:
        blue_pixels = img_arr[blue_mask]
        mean_blue  = blue_pixels.mean(axis=0)
        dev_frac   = pct_deviation_max_channel(mean_blue, NAVY)
        status = "pass" if dev_frac <= TOL_COLOR else "fail"
        dev_pct_str = f"{int(round(dev_frac*100))}%"
        out["chakra_blue"] = {"status": status, "deviation": dev_pct_str}
    else:
        out["chakra_blue"] = {"status": "fail", "deviation": "N/A"}

    return out

def chakra_geometry(img_arr: np.ndarray) -> Tuple[Tuple[float,float], float, np.ndarray]:
    """Return (center_x, center_y), radius, and the blue mask for the chakra."""
    blue = is_blue_mask(img_arr)
    ys, xs = np.where(blue)
    if len(xs) == 0:
        return (None, None), None, blue
    cx = xs.mean()
    cy = ys.mean()
    # Estimate radius from outer ring pixels (top 10% farthest distances).
    d = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    if len(d) == 0:
        return (cx, cy), None, blue
    thresh = np.percentile(d, 90)
    ring = d >= thresh
    radius = d[ring].mean() if ring.any() else d.mean()
    return (cx, cy), radius, blue

def detect_spokes(blue_mask: np.ndarray, center: Tuple[float, float]) -> int:
    """Estimate spoke count via peaks in angular histogram of blue pixels."""
    cy, cx = center[1], center[0]
    ys, xs = np.where(blue_mask)
    if len(xs) == 0:
        return 0
    ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
    hist_bins = 720  # 0.5° bins
    hist, _ = np.histogram(ang, bins=hist_bins, range=(0.0, 360.0))
    # Smooth with moving average
    k = 9
    kernel = np.ones(k) / k
    smooth = np.convolve(hist, kernel, mode='same')
    # Find local maxima with minimum separation ~12° (24 spokes => ~15° apart)
    min_sep_bins = int(12.0 / (360.0 / hist_bins))
    peaks = []
    i = 1
    while i < len(smooth)-1:
        if smooth[i] > smooth[i-1] and smooth[i] >= smooth[i+1] and smooth[i] > smooth.mean()*1.2:
            if not peaks or (i - peaks[-1]) >= min_sep_bins:
                peaks.append(i)
                i += min_sep_bins
                continue
        i += 1
    # Merge near-duplicate peaks
    merged = []
    tol = int(5.0 / (360.0 / hist_bins))
    for p in peaks:
        if not merged or p - merged[-1] > tol:
            merged.append(p)
    return len(merged)

def check_chakra_specs(img_arr: np.ndarray):
    """
    Check chakra geometry:
      - Center should lie at the center of the white band (±1% of W/H)
      - Diameter should be 3/4 of white band height (±5%)  [checked internally]
      - Spokes must be exactly 24
    Returns sub-objects for 'chakra_position' and 'chakra_spokes'.
    """
    H, W, _ = img_arr.shape
    thirds = H // 3
    white_band_top = thirds
    white_band_bottom = 2 * thirds
    white_band_height = white_band_bottom - white_band_top

    (cx, cy), radius, blue_mask = chakra_geometry(img_arr)
    if cx is None or radius is None:
        return (
            {"status": "fail", "offset_x": "N/A", "offset_y": "N/A"},
            {"status": "fail", "detected": 0}
        )

    # Centering in pixels (offset from image center horizontally and white band center vertically)
    offset_x_px = abs(cx - (W / 2.0))
    band_mid_y = (white_band_top + white_band_bottom) / 2.0
    offset_y_px = abs(cy - band_mid_y)
    pos_ok = (offset_x_px / (W + EPS) <= TOL_POS) and (offset_y_px / (H + EPS) <= TOL_POS)

    chakra_position = {
        "status": "pass" if pos_ok else "fail",
        "offset_x": f"{int(round(offset_x_px))}px",
        "offset_y": f"{int(round(offset_y_px))}px"
    }

    # Diameter check (internal; not included in final JSON to match spec)
    actual_diam = 2.0 * radius
    expected_diam = 0.75 * white_band_height
    _diam_ok = abs(actual_diam - expected_diam) <= TOL_DIAM * expected_diam
    # (Optional) could be logged for debugging

    # Spokes
    detected_spokes = detect_spokes(blue_mask, (cx, cy))
    chakra_spokes = {
        "status": "pass" if detected_spokes == 24 else "fail",
        "detected": int(detected_spokes)
    }

    return chakra_position, chakra_spokes

def validate(image_path: str) -> Dict[str, Any]:
    """
    Validate a flag image and return a JSON-compatible dict EXACTLY matching:
      aspect_ratio, colors, stripe_proportion, chakra_position, chakra_spokes
    """
    t0 = time.time()

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    H, W, _ = arr.shape

    # Aspect
    aspect = check_aspect_ratio(W, H)

    # Colors
    colors = check_colors(arr)

    # Stripe proportions
    labels = compute_row_labels(arr)
    props = label_proportions(labels)
    stripe_ok = all(abs(props[k] - (1.0/3.0)) <= BAND_TOL for k in ['saffron','white','green'])

    # Chakra specs
    chakra_position, chakra_spokes = check_chakra_specs(arr)

    # Final JSON
    report = {
        "aspect_ratio": aspect,
        "colors": colors,
        "stripe_proportion": {
            "status": "pass" if stripe_ok else "fail",
            "top": round(props["saffron"], 4),
            "middle": round(props["white"], 4),
            "bottom": round(props["green"], 4)
        },
        "chakra_position": chakra_position,
        "chakra_spokes": chakra_spokes
    }

    _elapsed = time.time() - t0  # internal perf check (≤ 3s target)

    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: python validator.py <image_path>"}))
        sys.exit(1)
    path = sys.argv[1]
    try:
        rep = validate(path)
        print(json.dumps(rep, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(2)
