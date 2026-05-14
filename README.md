# contour-extractor

CLI tool: photo with a person or animal in the foreground → clean line-art PNG with artist-style strokes. Uses a neural line-art model (controlnet-aux LineartDetector) for smooth continuous lines that match hand-drawn references, plus rembg for the outer silhouette. Five tunable detail tiers from `detailed` to `outline`.

## Pipeline

1. **`rembg`** (U2Net / ISNet) → alpha mask of the foreground subject
2. **Mask refinement** — morphological close + open for a clean foreground boundary
3. **`cv2.findContours`** on the mask → outer silhouette as a single uniform line
4. **Neural line-art** (controlnet-aux LineartDetector / LineartAnimeDetector) on the original image → artist-quality inner strokes
   - Fallback: Canny or XDoG (built-in) if neural deps aren't installed
5. Hard threshold + connected-component pruning → flat black strokes, no gradients
6. Composite as RGBA PNG, black strokes on transparent or opaque white

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/LyreZ61/contour-extractor.git
cd contour-extractor
uv venv --python 3.12

# CPU torch first (smaller, avoids CUDA bloat):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# everything else (rembg, controlnet-aux, opencv, etc.)
uv pip install -r requirements.txt
```

First run downloads model weights to `~/.u2net/` and the HuggingFace cache (~600 MB total). Subsequent runs are local.

If you want the lightweight, no-torch path: skip the neural line-art install and use `--style canny` or `--style sketch`.

## Usage

```bash
.venv/bin/python contour.py input.jpg output.png
```

### Detail tiers

`--detail-level` picks one of five presets that step from dense artist-style strokes down to a pure silhouette:

```bash
# dense artist-style strokes (default)
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg --detail-level detailed

# fewer strokes, still recognisable features
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg --detail-level medium

# key features only — eyes, mouth, hair outline
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg --detail-level simple

# outline + minimal feature hints
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg --detail-level minimal

# pure silhouette, no inner detail
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg --detail-level outline
```

### Recipes

```bash
# default — neural line-art, transparent background, human subject
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg

# opaque white background (printable)
.venv/bin/python contour.py photo.jpg sketch.png --model u2net_human_seg --background white

# animal photo
.venv/bin/python contour.py cat.jpg cat_line.png --model isnet-general-use

# higher-resolution neural detection (slower, more fine detail)
.venv/bin/python contour.py photo.jpg out.png --lineart-resolution 768

# fall back to classic Canny (no torch required)
.venv/bin/python contour.py photo.jpg out.png --style canny
```

### Batch over a folder

```bash
./batch.sh
```

Reads `examples/test/*.jpg`, writes transparent + white variants to `examples/test_out/`.

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `u2net` | `u2net_human_seg` for humans, `isnet-general-use` strong all-rounder |
| `--style` | `lineart` | `lineart` (neural, default), `lineart-anime` (cleaner/simpler), `canny`, `sketch` |
| `--detail-level` | — | `detailed` / `medium` / `simple` / `minimal` / `outline` (overrides individual flags) |
| `--lineart-resolution` | `512` | working resolution for the neural model (256/512/768) |
| `--lineart-coarse` | off | coarse mode of `LineartDetector` (only `--style lineart`) |
| `--background` | `transparent` | `transparent` or `white` |
| `--outer-thickness` | `1` | silhouette line width px (match `--inner-thickness` for uniform stroke) |
| `--inner-thickness` | `1` | inner edge dilation px |
| `--binarize-threshold` | `20` | inner-edge threshold (0–255). Lower = more strokes kept |
| `--inner-close` | `0` | morphological close radius on inner edges (reconnects gaps) |
| `--min-component-size` | `4` | drop stroke components smaller than N pixels |
| `--canny-low` / `--canny-high` | `30` / `80` | Canny thresholds (only `--style canny`) |
| `--alpha-matting` | off | sharper fur/hair contour from rembg |
| `--mask-close` / `--mask-open` | `4` / `2` | mask morphological cleanup radii |
| `--no-inner` | off | silhouette only |

## Tuning cheatsheet

- **Try detail tiers first** — `--detail-level detailed/medium/simple/minimal/outline` covers most needs
- **Want thicker lines** → raise both `--outer-thickness` and `--inner-thickness` together (e.g. `2 2`)
- **Output too dense** → step down a tier, or raise `--binarize-threshold`
- **Output too sparse** → step up a tier, or lower `--binarize-threshold`
- **Outer contour ragged on fuzzy subjects (hair, fur)** → enable `--alpha-matting`
- **Wrong subject detected** → try `--model isnet-general-use` or `--model silueta`
- **No torch / want fast** → `--style canny` (classic, no neural model)

## Test set

`examples/test/urls.txt` lists 15 [Unsplash](https://unsplash.com) photos (portraits, cats, dogs, horses, wildlife).

```bash
./scripts/fetch_test_images.sh
./batch.sh
```

Showcase renderings live in `examples/test_out/`. Photos used under the [Unsplash License](https://unsplash.com/license).

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
