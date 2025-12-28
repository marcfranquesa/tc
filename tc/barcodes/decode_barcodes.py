from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .code128_decoder import Code128Result, decode_code128_from_binary_rows


@dataclass(frozen=True)
class CropResult:
    crop_bgr: np.ndarray
    crop_mask: np.ndarray  # uint8 0/255
    angle_deg: float
    bbox_xyxy: Tuple[int, int, int, int]  # in original image


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1


def _rotate_crop_and_mask(
    crop: np.ndarray, mask: np.ndarray, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot_img = cv2.warpAffine(
        crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    rot_m = cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rot_img, rot_m


def extract_and_deskew_from_mask(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    *,
    pad_frac: float = 0.55,
    quiet_x_frac: float = 1.0,
    quiet_y_frac: float = 0.45,
) -> CropResult:
    """
    Extract a padded crop around the mask and rotate so the barcode's long axis is horizontal.
    """
    x0, y0, x1, y1 = _mask_bbox(mask_bool)
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    pad_x = int(max(10, pad_frac * w))
    pad_y = int(max(10, pad_frac * h))

    H, W = image_bgr.shape[:2]
    x0p = max(0, x0 - pad_x)
    y0p = max(0, y0 - pad_y)
    x1p = min(W - 1, x1 + pad_x)
    y1p = min(H - 1, y1 + pad_y)

    crop = image_bgr[y0p : y1p + 1, x0p : x1p + 1].copy()
    mask_u8 = (mask_bool[y0p : y1p + 1, x0p : x1p + 1].astype(np.uint8) * 255).copy()

    # Estimate rotation from the mask contour.
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        angle = 0.0
    else:
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), (rw, rh), theta = cv2.minAreaRect(cnt)
        # OpenCV: theta is in [-90, 0). We want the long axis horizontal.
        if rw < rh:
            theta = theta + 90.0
        # Deskew: rotate by -theta.
        angle = -float(theta)

    rot_img, rot_mask = _rotate_crop_and_mask(crop, mask_u8, angle)

    # Tight crop around rotated mask, with anisotropic quiet-zone padding.
    ys, xs = np.where(rot_mask > 0)
    if ys.size == 0:
        final = rot_img
        final_mask = rot_mask
    else:
        ry0, ry1 = int(ys.min()), int(ys.max())
        rx0, rx1 = int(xs.min()), int(xs.max())
        rh = ry1 - ry0 + 1
        rw = rx1 - rx0 + 1
        qx = int(max(20, quiet_x_frac * rw))
        qy = int(max(10, quiet_y_frac * rh))
        ry0 = max(0, ry0 - qy)
        ry1 = min(rot_img.shape[0] - 1, ry1 + qy)
        rx0 = max(0, rx0 - qx)
        rx1 = min(rot_img.shape[1] - 1, rx1 + qx)
        final = rot_img[ry0 : ry1 + 1, rx0 : rx1 + 1]
        final_mask = rot_mask[ry0 : ry1 + 1, rx0 : rx1 + 1]

    return CropResult(
        crop_bgr=final,
        crop_mask=final_mask,
        angle_deg=angle,
        bbox_xyxy=(x0p, y0p, x1p, y1p),
    )


def preprocess_for_1d_decode(
    crop_bgr: np.ndarray, *, scale: float = 3.0, clahe: bool = True
) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(
            gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = c.apply(gray)

    # Mild denoise + sharpen
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # Otsu binarization
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure background is white (255) near borders.
    h, w = bw.shape
    border = np.concatenate(
        [bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]],
        axis=0,
    )
    if float(np.mean(border)) < 127.0:
        bw = 255 - bw

    # Clean tiny specks without changing bar widths too much.
    bw = cv2.medianBlur(bw, 3)
    return bw


def preprocess(
    crop_bgr: np.ndarray, *, scale: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        g = cv2.resize(g, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g, bw


def _prefer_vertical_bars(gray_or_bw: np.ndarray) -> int:
    """
    Return rot90 k in {0,1} so that after rotation the barcode bars are more likely vertical.

    Heuristic: compare mean absolute Sobel gradients; vertical bars => stronger x-gradient.
    """
    if gray_or_bw.ndim != 2:
        raise ValueError("_prefer_vertical_bars expects 2D image")
    img = gray_or_bw
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    def score(im: np.ndarray) -> float:
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
        sx = float(np.mean(np.abs(gx)))
        sy = float(np.mean(np.abs(gy)))
        return sx - sy

    s0 = score(img)
    s1 = score(np.ascontiguousarray(np.rot90(img, 1)))
    return 0 if s0 >= s1 else 1


def decode_code128_best_effort(
    bgr: np.ndarray,
) -> Tuple[Optional[Code128Result], np.ndarray, int]:
    """
    Decode Code128 from an already-extracted barcode view.
    Returns (result, bw_used, rot90_k) where rot90_k is in {0,1,2,3}.
    """
    # Prefer Otsu on CLAHE grayscale; it tends to preserve bar edges.
    g0, bw0 = preprocess(bgr, scale=1.0)
    k_pref = _prefer_vertical_bars(g0)
    bw0 = np.ascontiguousarray(np.rot90(bw0, k_pref))

    for k in (0, 1, 2, 3):
        bw = np.ascontiguousarray(np.rot90(bw0, k))
        res = decode_code128_from_binary_rows(
            bw, max_rows=81, row_band=0.8, min_runs=20, band_half=3
        )
        if res is not None:
            return res, bw, (k_pref + k) % 4
    return None, bw0, k_pref


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as [tl, tr, br, bl]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_from_mask(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    *,
    scale_w: float = 1.6,
    scale_h: float = 1.2,
    out_scale: float = 6.0,
    min_out_hw: Tuple[int, int] = (160, 120),
) -> np.ndarray:
    """
    Perspective-rectify a region around the mask using an expanded minAreaRect.
    This is still a mild warp (mostly affine for small tilts) but is critical for
    curved/angled surfaces where pure rotation loses quiet zones.
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255).copy()
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image_bgr.copy()
    cnt = max(cnts, key=cv2.contourArea)

    (cx, cy), (rw, rh), theta = cv2.minAreaRect(cnt)
    rw2 = float(rw) * float(scale_w)
    rh2 = float(rh) * float(scale_h)
    rect2 = ((cx, cy), (rw2, rh2), float(theta))
    box = cv2.boxPoints(rect2)
    src = _order_quad_points(box)

    out_w = int(max(min_out_hw[0], rw2 * float(out_scale)))
    out_h = int(max(min_out_hw[1], rh2 * float(out_scale)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(
        image_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warp
