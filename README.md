# contour-extractor

CLI tool: photo with a person or animal in the foreground → transparent (or white) PNG with a pencil-sketch outline plus inner detail strokes.

## Pipeline

1. **`rembg`** (U2Net / ISNet) → alpha mask of the foreground subject
2. **`cv2.findContours`** on the mask → outer silhouette (thick stroke)
3. **XDoG** (Extended Difference of Gaussians, default) or **Canny** edges on the masked subject → pencil-style inner strokes, restricted to the subject's interior (mask erosion avoids double-drawing the rim)
4. Optional per-image **min-max normalization** + **gamma** post-process push faint strokes up so low-contrast subjects (white fur, soft studio light) still produce visible inner detail
5. Composite both layers into an RGBA PNG: black strokes on transparent or opaque white background

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/LyreZ61/contour-extractor.git
cd contour-extractor
uv venv --python 3.12
uv pip install -r requirements.txt
```

First run downloads the rembg model (~170 MB) to `~/.u2net/`.

## Usage

```bash
.venv/bin/python contour.py input.jpg output.png
```

### Recipes

```bash
# default — XDoG sketch style, transparent background, human subject
.venv/bin/python contour.py photo.jpg out.png --model u2net_human_seg

# white background (opaque white, ready to print or use directly)
.venv/bin/python contour.py photo.jpg sketch.png --model u2net_human_seg --background white

# animal photo
.venv/bin/python contour.py cat.jpg cat_line.png --model isnet-general-use

# silhouette only (no inner detail)
.venv/bin/python contour.py photo.jpg sil.png --no-inner --outer-thickness 6

# bolder strokes
.venv/bin/python contour.py photo.jpg bold.png --model u2net_human_seg --inner-thickness 2 --gamma 0.5

# classic Canny edges instead of XDoG sketch
.venv/bin/python contour.py photo.jpg edges.png --style canny --canny-low 40 --canny-high 120
```

### Batch over a folder

```bash
./batch.sh
```

Reads `examples/test/*.jpg`, writes `examples/test_out/<name>_transparent.png` and `<name>_white.png`. Auto-selects the rembg model from the filename (`*portrait*` → `u2net_human_seg`, otherwise `isnet-general-use`).

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `u2net` | `u2net_human_seg` for humans, `isnet-general-use` strong all-rounder, `silueta` lightest |
| `--style` | `sketch` | `sketch` (XDoG pencil strokes) or `canny` (sharp edges) |
| `--background` | `transparent` | `transparent` or `white` |
| `--outer-thickness` | `4` | silhouette line width px |
| `--inner-thickness` | `1` | inner edge dilation px (1 = thin, 2-3 = bold) |
| `--xdog-sigma` | `0.8` | XDoG base sigma — smaller = finer detail |
| `--xdog-k` | `1.6` | ratio of second sigma — typ. 1.4–2.0 |
| `--xdog-tau` | `0.99` | second Gaussian weight (closer to 1 = thinner strokes) |
| `--xdog-epsilon` | `0.005` | XDoG threshold — higher = more strokes |
| `--xdog-phi` | `20.0` | edge sharpness — higher = harder edges |
| `--no-xdog-normalize` | off | disable per-image min-max rescaling (keeps relative stroke strength across photos) |
| `--gamma` | `0.7` | post-process gamma on inner edges (<1 boosts faint strokes, 1.0 = off) |
| `--canny-low` | `60` | Canny lower threshold (only for `--style canny`) |
| `--canny-high` | `160` | Canny upper threshold |
| `--clahe-clip` | `3.5` | local contrast boost before edge detection (0 = off) |
| `--smooth` | `1` | Gaussian blur radius on mask before contour |
| `--erode` | `3` | shrink mask before inner edges, avoids double-drawing the rim |
| `--no-inner` | off | silhouette only |

## Tuning

- **Lines too pale** → lower `--gamma` (e.g. `0.5`) or raise `--inner-thickness` to `2`
- **Output too dark / too dense** → raise `--gamma` (e.g. `0.9`) or raise `--xdog-epsilon`
- **Silhouette jagged** → raise `--smooth` (e.g. `3`)
- **Subject mis-detected** → pick a different `--model` (try `isnet-general-use` first, fall back to `silueta` for fast or unusual subjects)
- **Rim doubled** → raise `--erode` (e.g. `5`)
- **Subject has lots of fine texture you want to skip** → raise `--xdog-sigma` (e.g. `1.2`)

## Test set

`examples/test/urls.txt` lists 15 [Unsplash](https://unsplash.com) photos (5 portraits, 3 cats, 3 dogs, 2 horses, 2 wildlife). Pull them with:

```bash
./scripts/fetch_test_images.sh
./batch.sh
```

Renderings in `examples/test_out/`. Used under the [Unsplash License](https://unsplash.com/license) (free, no attribution required).

## Repo layout

```
contour.py                    main CLI
batch.sh                      batch script over examples/test/
requirements.txt              Python deps
scripts/fetch_test_images.sh  download the 15-image Unsplash test set
examples/test/urls.txt        source URLs
examples/test_out/            rendered showcase outputs (committed)
```

## License

[MIT](LICENSE).
