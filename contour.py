#!/usr/bin/env python3
"""
Contour extractor: subject (person/animal) → transparent PNG line drawing.

Pipeline:
  1. rembg (U2Net) → alpha mask of foreground subject
  2. findContours on mask → outer silhouette (thick stroke)
  3. Inner edges via Canny or XDoG (sketch-style) on masked subject
  4. Composite onto transparent or white RGBA canvas
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def foreground_mask(img_bgr: np.ndarray, model: str) -> np.ndarray:
    """Run rembg and return uint8 alpha mask (0..255)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    session = new_session(model)
    cutout = remove(pil, session=session)
    rgba = np.array(cutout.convert("RGBA"))
    return rgba[:, :, 3]


def outer_contour_layer(mask: np.ndarray, thickness: int, smooth: int) -> np.ndarray:
    """Return uint8 single-channel image, 255 where contour line is drawn."""
    m = mask.copy()
    if smooth > 0:
        k = smooth * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    _, binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    # close small holes so contour is one clean curve
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    layer = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(layer, contours, -1, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return layer


def _preprocess_gray(img_bgr: np.ndarray, clahe_clip: float) -> np.ndarray:
    """Grayscale + local contrast (CLAHE) + edge-preserving smoothing."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def _canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, low, high)


def _xdog_edges(
    gray: np.ndarray,
    sigma: float,
    k: float,
    tau: float,
    epsilon: float,
    phi: float,
    normalize: bool,
) -> np.ndarray:
    """Extended Difference of Gaussians → pencil-sketch style continuous strokes.

    Reference: Winnemoeller et al., XDoG: an eXtended difference-of-Gaussians.
    Returns uint8 image where strokes are 255 on 0 background.
    When `normalize` is True, stretches output to full 0–255 range so low-contrast
    subjects produce strokes of comparable boldness to high-contrast subjects.
    """
    g = gray.astype(np.float32) / 255.0
    blur1 = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)
    blur2 = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma * k)
    dog = blur1 - tau * blur2
    # soft threshold (sketch curve)
    sketch = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
    sketch = np.clip(sketch, 0.0, 1.0)
    # invert so strokes are bright on dark
    strokes = (1.0 - sketch) * 255.0

    if normalize:
        lo, hi = float(strokes.min()), float(strokes.max())
        # only rescale when there's signal; avoid blowing up nearly-uniform images
        if hi - lo > 5.0:
            strokes = (strokes - lo) * (255.0 / (hi - lo))
            strokes = np.clip(strokes, 0.0, 255.0)
    return strokes.astype(np.uint8)


def inner_edges_layer(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    style: str,
    canny_low: int,
    canny_high: int,
    xdog_sigma: float,
    xdog_k: float,
    xdog_tau: float,
    xdog_epsilon: float,
    xdog_phi: float,
    xdog_normalize: bool,
    gamma: float,
    clahe_clip: float,
    thickness: int,
    erode: int,
) -> np.ndarray:
    """Return uint8 single-channel edge image inside the foreground only."""
    gray = _preprocess_gray(img_bgr, clahe_clip)

    if style == "canny":
        edges = _canny_edges(gray, canny_low, canny_high)
    elif style == "sketch":
        edges = _xdog_edges(
            gray, xdog_sigma, xdog_k, xdog_tau, xdog_epsilon, xdog_phi, xdog_normalize
        )
    else:
        raise ValueError(f"unknown style: {style}")

    # gamma post-process: gamma<1 boosts faint strokes (darker output), >1 fades them
    if gamma > 0 and abs(gamma - 1.0) > 1e-3:
        norm = edges.astype(np.float32) / 255.0
        norm = np.power(norm, gamma)
        edges = (norm * 255.0).astype(np.uint8)

    # restrict to interior of subject (avoid double-drawing outer edge)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        binary = cv2.erode(binary, kernel)
    edges = cv2.bitwise_and(edges, edges, mask=binary)

    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        edges = cv2.dilate(edges, kernel)
    return edges


def composite_rgba(outer: np.ndarray, inner: np.ndarray, background: str) -> np.ndarray:
    """Black lines combined. background = 'transparent' or 'white'."""
    h, w = outer.shape
    combined = cv2.max(outer, inner)  # 0..255 line strength

    if background == "transparent":
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 3] = combined  # alpha = line strength, RGB stays 0 (black)
        return rgba
    if background == "white":
        # black strokes (line strength → darkness) on opaque white
        line = combined.astype(np.float32) / 255.0
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        rgb = (rgb.astype(np.float32) * (1.0 - line[..., None])).astype(np.uint8)
        rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
        return rgba
    raise ValueError(f"unknown background: {background}")


def extract_contour(
    input_path: Path,
    output_path: Path,
    *,
    model: str = "u2net",
    style: str = "sketch",
    background: str = "transparent",
    outer_thickness: int = 4,
    inner_thickness: int = 1,
    canny_low: int = 60,
    canny_high: int = 160,
    xdog_sigma: float = 0.8,
    xdog_k: float = 1.6,
    xdog_tau: float = 0.99,
    xdog_epsilon: float = 0.005,
    xdog_phi: float = 20.0,
    xdog_normalize: bool = True,
    gamma: float = 0.7,
    clahe_clip: float = 3.5,
    smooth: int = 1,
    erode: int = 3,
) -> None:
    img = load_image_bgr(input_path)
    mask = foreground_mask(img, model)
    outer = outer_contour_layer(mask, outer_thickness, smooth)
    if inner_thickness <= 0:
        inner = np.zeros_like(mask, dtype=np.uint8)
    else:
        inner = inner_edges_layer(
            img,
            mask,
            style=style,
            canny_low=canny_low,
            canny_high=canny_high,
            xdog_sigma=xdog_sigma,
            xdog_k=xdog_k,
            xdog_tau=xdog_tau,
            xdog_epsilon=xdog_epsilon,
            xdog_phi=xdog_phi,
            xdog_normalize=xdog_normalize,
            gamma=gamma,
            clahe_clip=clahe_clip,
            thickness=inner_thickness,
            erode=erode,
        )
    rgba = composite_rgba(outer, inner, background)
    Image.fromarray(rgba, mode="RGBA").save(str(output_path), format="PNG")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract contour + inner edges from subject image → transparent PNG.",
    )
    p.add_argument("input", type=Path, help="input image (jpg/png/...)")
    p.add_argument("output", type=Path, help="output PNG path")
    p.add_argument(
        "--model",
        default="u2net",
        choices=["u2net", "u2netp", "u2net_human_seg", "isnet-general-use", "isnet-anime", "silueta"],
        help="rembg model (u2net_human_seg best for humans, isnet-general-use strong all-rounder)",
    )
    p.add_argument(
        "--style",
        default="sketch",
        choices=["sketch", "canny"],
        help="inner edge style: 'sketch' (XDoG, pencil-style strokes) or 'canny' (sharp edges)",
    )
    p.add_argument(
        "--background",
        default="transparent",
        choices=["transparent", "white"],
        help="output background",
    )
    p.add_argument("--outer-thickness", type=int, default=4, help="silhouette line width px")
    p.add_argument("--inner-thickness", type=int, default=1, help="inner edge line width px")
    p.add_argument("--canny-low", type=int, default=60, help="Canny lower threshold (canny style)")
    p.add_argument("--canny-high", type=int, default=160, help="Canny upper threshold (canny style)")
    p.add_argument("--xdog-sigma", type=float, default=0.8, help="XDoG base sigma — smaller = finer detail")
    p.add_argument("--xdog-k", type=float, default=1.6, help="XDoG sigma ratio (typ. 1.4 – 2.0)")
    p.add_argument("--xdog-tau", type=float, default=0.99, help="XDoG second Gaussian weight (closer to 1 = thinner strokes)")
    p.add_argument("--xdog-epsilon", type=float, default=0.005, help="XDoG threshold — higher = more strokes")
    p.add_argument("--xdog-phi", type=float, default=20.0, help="XDoG sharpness — higher = harder edges")
    p.add_argument(
        "--no-xdog-normalize",
        action="store_true",
        help="disable per-image min-max normalization on XDoG output",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="post-process gamma on inner edges (<1 boosts faint strokes, >1 fades, 1 = off)",
    )
    p.add_argument("--clahe-clip", type=float, default=3.5, help="CLAHE clip limit (0 = disable)")
    p.add_argument("--smooth", type=int, default=1, help="mask Gaussian blur radius (0 = off)")
    p.add_argument("--erode", type=int, default=3, help="erode mask before inner edges to skip rim")
    p.add_argument("--no-inner", action="store_true", help="silhouette only, skip inner edges")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    extract_contour(
        args.input,
        args.output,
        model=args.model,
        style=args.style,
        background=args.background,
        outer_thickness=args.outer_thickness,
        inner_thickness=0 if args.no_inner else args.inner_thickness,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        xdog_sigma=args.xdog_sigma,
        xdog_k=args.xdog_k,
        xdog_tau=args.xdog_tau,
        xdog_epsilon=args.xdog_epsilon,
        xdog_phi=args.xdog_phi,
        xdog_normalize=not args.no_xdog_normalize,
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        smooth=args.smooth,
        erode=args.erode,
    )
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
