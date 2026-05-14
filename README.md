# contour-extractor

CLI tool: photo with a person or animal in the foreground → clean line-art PNG with a uniform black outline plus inner detail strokes. Transparent or opaque white background.

## Pipeline

1. **`rembg`** (U2Net / ISNet) → alpha mask of the foreground subject
2. **Mask refinement** — morphological close (fill holes) + open (remove specks)
3. **`cv2.findContours`** on the refined mask → outer silhouette as a single uniform black line
4. **`cv2.pyrMeanShiftFiltering`** on the original image → flattens fabric, fur and skin texture into uniform regions
5. **Canny edges** (default) or **XDoG** on the flattened image → inner detail lines
6. Hard threshold + small-component pruning → no gradients, no noise specks
7. Composite as RGBA PNG, black strokes on transparent or opaque white

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/LyreZ61/contour-extractor.git
cd contour-extractor
uv venv --python 3.12
uv pip install -r requirements.txt
```

First run downloads a rembg model (~170 MB) to `~/.u2net/`.

## Usage

```bash
.venv/bin/python contour.py input.jpg output.png
```

### Recipes

```bash
# default — Canny line-art, transparent background, human subject
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg

# opaque white background (ready for print or colouring)
.venv/bin/python contour.py photo.jpg sketch.png --model u2net_human_seg --background white

# animal photo
.venv/bin/python contour.py cat.jpg cat_line.png --model isnet-general-use

# very textured fabric (cable-knit, dense pattern) — push flattening up
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg --flatten 50 --min-component-size 80

# silhouette only (no inner detail)
.venv/bin/python contour.py photo.jpg sil.png --no-inner --outer-thickness 6

# XDoG pencil-sketch style instead of clean line-art
.venv/bin/python contour.py photo.jpg sketch.png --style sketch

# fuzzy subject (lots of hair/fur) — enable alpha matting for sharper edge
.venv/bin/python contour.py portrait.jpg out.png --model u2net_human_seg --alpha-matting
```

### Batch over a folder

```bash
./batch.sh
```

Reads `examples/test/*.jpg`, writes both transparent and white versions to `examples/test_out/`. Auto-selects the rembg model from the filename (`*portrait*` → `u2net_human_seg`, otherwise `isnet-general-use`).

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `u2net` | `u2net_human_seg` for humans, `isnet-general-use` strong all-rounder |
| `--style` | `canny` | `canny` (clean line-art) or `sketch` (XDoG pencil strokes) |
| `--background` | `transparent` | `transparent` or `white` |
| `--outer-thickness` | `1` | silhouette line width px (match `--inner-thickness` for uniform stroke) |
| `--inner-thickness` | `1` | inner edge dilation px |
| `--canny-low` | `30` | Canny lower threshold |
| `--canny-high` | `80` | Canny upper threshold |
| `--flatten` | `0` | `pyrMeanShift` texture-flattening strength (0 = off, keeps detail; 30–50 to wash out fabric) |
| `--bilateral-strength` | `40` | final smoothing pass strength |
| `--clahe-clip` | `2.0` | local contrast boost (0 = off) |
| `--binarize-threshold` | `20` | inner-edge threshold (0–255) |
| `--min-component-size` | `4` | drop stroke components smaller than N pixels |
| `--smooth` | `1` | Gaussian blur radius on the mask before contour |
| `--erode` | `2` | shrink mask before inner edges to skip the rim |
| `--alpha-matting` | off | sharper fur/hair contour (slower, can pick up background blobs on cluttered photos) |
| `--mask-close` | `4` | morphological close radius on the mask |
| `--mask-open` | `2` | morphological open radius on the mask |
| `--no-inner` | off | silhouette only |

## Tuning cheatsheet

- **Want thicker lines** → raise both `--outer-thickness` and `--inner-thickness` together (e.g. `2 2`); keep them equal so strokes stay uniform
- **Output too dense / cluttered** → raise `--canny-low` (e.g. `60`) and `--canny-high` (e.g. `150`), or raise `--flatten` (e.g. `30`) to wash texture
- **Output too sparse / missing detail** → lower `--canny-low` (e.g. `15`) and `--canny-high` (e.g. `50`)
- **Speckled fabric / wool noise** → raise `--flatten` (e.g. `50`) and `--min-component-size` (e.g. `40`)
- **Outer contour ragged on fuzzy subjects (hair, fur)** → enable `--alpha-matting`
- **Silhouette jagged** → raise `--smooth` (e.g. `3`)
- **Wrong subject detected** → try `--model isnet-general-use` (general) or `--model silueta` (lightweight)
- **Mask has small holes / detached blobs** → raise `--mask-close` / `--mask-open` (e.g. `6` / `4`)
- **Want pencil-sketch style instead of clean line-art** → `--style sketch`

## Test set

`examples/test/urls.txt` lists 15 [Unsplash](https://unsplash.com) photos (portraits, cats, dogs, horses, wildlife). Pull and render with:

```bash
./scripts/fetch_test_images.sh
./batch.sh
```

Rendered showcase outputs are committed to `examples/test_out/`. Photos used under the [Unsplash License](https://unsplash.com/license).

## Repo layout

```
contour.py                    main CLI
batch.sh                      batch driver over examples/test/
requirements.txt              Python deps
scripts/fetch_test_images.sh  download the 15-image Unsplash test set
examples/test/urls.txt        source URLs
examples/test_out/            rendered showcase outputs (committed)
```

## License

[MIT](LICENSE).
