# slicereg

Load a slide scan into napari, auto-detect and crop the brain sections, separate
channels, align each section to the Allen CCF, click-count cells, and visualise the
results in 3D with brainrender.

## Installation

Requires Python 3.10-3.12 (napari and brainrender may not yet support newer versions)
from [python.org](https://www.python.org/downloads/windows/), not Anaconda. Anaconda's
Python ships an old C++ runtime that stops PyQt6 from loading
(`qtpy.QtBindingsNotFoundError: No Qt bindings could be found`), even inside a venv.
Run `py -0p` to list installed Pythons.

From a PowerShell prompt in the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. If your prompt shows `(base)`,
run `conda deactivate` first so the Anaconda environment isn't mixed in.

The last step installs the `slicereg` command. Without it you can use
`python -m slicereg` instead everywhere below.

The first time a project is aligned, the BrainGlobe atlas (`allen_mouse_25um` by
default) is downloaded to `~/.brainglobe`, which needs an internet connection.

### Optional: DeepSlice

[DeepSlice](https://github.com/PolarBean/DeepSlice) predicts a section's position in
the Allen mouse atlas, which gives a quick initial alignment to refine by hand. It
needs TensorFlow and older versions of some packages, so it gets its own environment
(`.venv-deepslice` in the repository folder, about 2 GB including model weights). With
the main environment active, run once:

```powershell
slicereg deepslice-setup
```

## Launching

Activate the environment first (`.\.venv\Scripts\Activate.ps1`), then open a slide scan:

```powershell
slicereg open "C:\path\to\slide.tif"
```

This creates a project folder `slide_project` next to the TIFF and opens the napari
window. Running the same command again (or passing the project folder itself)
reopens the project where you left off.

### A folder of section images

If each section was imaged separately (e.g. Zeiss `.lsm` tile scans), open the
folder instead:

```powershell
slicereg open "E:\Histology\Zeiss\TJO_Optotag_16"
```

The first time, this writes `slicereg_folder.json` into the image folder and creates
`TJO_Optotag_16_project` next to it. The config says which files to use and what
their names mean:

| Key | Default | Meaning |
| --- | --- | --- |
| `pattern` | `c(?P<col>\d+)_r(?P<row>\d+).*_MIP\.lsm$` | Regular expression for the files to use; the `col` and `row` groups give the section's position on the slide |
| `order` | `column` | Section order for Prev/Next and AP guesses: `column` = c1_r1, c1_r2, ...; `row` = c1_r1, c2_r1, ... |
| `channels` | rfp (red), nissl (blue) | Name and napari colour of each channel, in file order |
| `align_channel` | `nissl` | Channel DeepSlice sees |
| `auto_crop` | `true` | Crop each image to its main section and blank bits of neighbouring sections |
| `pixel_um` | `null` | Pixel size in microns; `null` reads it from the file |
| `hemispheres` | `hemispheres.csv` | Optional CSV in the image folder saying which hemisphere is on each image's left and right (see below) |
| `atlas`, `section_spacing_um` | `allen_mouse_25um`, 80 | Used when the project is created |

The slide view then shows every section at true size in its column/row, cropped to
its tissue and with each section's brightness stretched separately so sections
imaged with different settings look alike (the slice view itself shows the raw
data). Click a section to select it (its box turns cyan) and double-click it, or
use *Open selected section*, to align and count it exactly as above. Z-stacks are max-projected, and mosaics
that were saved as separate tiles are stitched from the tile positions in the file.
Click *Rescan folder* after editing the config or adding images; existing sections
keep their alignments and cells.

**Hemispheres.** Sections can land on the slide either way up, so some images show the
right hemisphere on the left and others the left. List them in `hemispheres.csv` in
the image folder, with the hemisphere on the image's left in `x1` and on its right in
`x2` (`L`/`R`; names are matched ignoring case, `_` and `-`):

```text
slice,x1,x2
c4_r1,L,R
c4_r2,R,L
```

Every section is then stored with the right hemisphere on the image's left, like the
atlas: `L,R` sections are mirrored left-right when imported or when you click *Rescan
folder*. Alignments, landmarks and cells are mirrored with the image, so the atlas
outline stays on the same tissue. Where an existing alignment had the atlas's
hemispheres the wrong way round, the atlas is mirrored too. Its outline doesn't move,
but left and right swap and the cells' hemisphere follows. For sections in the file,
*Mirror atlas left/right* is locked because the file already decides it. Export again
after a rescan that mirrors sections.

Options for `open`:

| Option | Meaning |
| --- | --- |
| `--project DIR` | Put the project folder somewhere else |
| `--atlas NAME` | BrainGlobe atlas for a new project, e.g. `allen_mouse_10um` |
| `--spacing UM` | Section spacing (thickness) in microns, used to guess an unaligned section's AP position from its aligned neighbours in slide order |

## Workflow

1. **Slide view.** Detected sections are shown as yellow boxes. Drag to move or
   resize them, draw missing ones with the rectangle tool, and delete wrong ones with
   the Delete key. Use *Re-detect sections* to start over.
2. Click **Crop & save sections** to write each section to the project folder.
3. Double-click a section (or pick it in the list and click *Open selected section*).
4. **Display** panel (above the tabs, shared by both): toggle *Show atlas outlines*
   and *Show region colours*, and pick between crisp *Smooth outlines* off (the raw
   atlas voxel edges) and on (a sub-pixel contour fit through each region boundary).
   When smoothing is on, *Smoothing* controls how much the contour is blurred first
   and *Line width* / *Outline colour...* control how it's drawn. Check
   *Show average template side by side* to add a second panel next to the section
   showing the atlas's average template (the Allen CCF's averaged reference brain,
   or the MRI average for MRI-based atlases) with the same outlines, panned and
   zoomed together with the section; hovering either panel shows a crosshair at the
   matching spot in the other. These settings are saved per project.
5. **1. Align to atlas** tab: set the AP position, tilt, rotation, scale and centre
   until the atlas outline matches the tissue. *Predict alignment with DeepSlice*
   (Allen mouse atlases only, see [Optional: DeepSlice](#optional-deepslice)) fills
   all of these in from the section image in about 10 s (it is shown only the tissue
   inside the section's mask, so neighbouring sections don't confuse it); your *Mirror atlas
   left/right* setting is kept, since DeepSlice can't tell the hemispheres apart. If
   the guess is off on a fluorescence image, tick *Invert image* and run it again;
   *Undo DeepSlice* restores the previous alignment and landmarks. *Auto-fit to
   tissue outline* is a simpler starting point that only fits scale and centre;
   Shift+drag moves the atlas. *Landmark mode* lets you drag atlas features onto
   matching tissue features to warp the fit.
6. **2. Count cells** tab: choose or create a cell type, then click each cell in
   *Add cells* mode; Shift+drag moves the view without adding a cell, and the scroll
   wheel zooms. Use *Select / delete* to remove marks. The *Cell marker size*
   slider changes how big the marks are drawn (this is just display, it doesn't
   affect the saved coordinates).
7. Click **Save (alignment + cells)**. Use *< Prev* / *Next >* to move between
   sections and *Slide* to return to the overview.
8. Back in the slide view, click **Plot AP & pitch per section** to compare the
   alignments with each other. The top plot shows each section's AP position (um) in
   slide order. Consecutive sections should step evenly, by the section thickness. The
   bottom plot shows each section's pitch, which should be a flat line because every
   section from one brain was cut at the same angle. Click a point to open that
   section, and use the toolbar to zoom or save the figure. Then click
   **Export all cells (CSV)** or
   **3D view in brainrender**. The 3D view button opens an options dialog: type the
   structures to show (see below), choose how many top regions by cell count to add,
   the structure opacity and hemisphere, the whole-brain opacity, the surface style
   (smooth by default; *Cartoon* is brainrender's flat outlined look), the starting
   view, the cell size (sphere radius in um, 25 by default; cells are always drawn
   matte, whatever the surface style), and whether to show section images, cells (per
   cell type), the whole brain, axes / scale bars (off by default), the small
   orientation brain in the corner and the title. The last choices are remembered.
   In the 3D window, buttons on the left snap the camera to a 3/4, front, back, left,
   right, top or bottom view. *Back, low* looks from behind and about 12 degrees
   below, aimed at the caudal end of IRt/PCRt, with the whole brain in frame. *Back,
   low, zoomed* uses the same angle but frames just IRt/PCRt. Both work whether or
   not those structures are shown. *Hide title* / *Show title* toggles the title, for
   example before saving. *Save image* asks where to save a PNG at twice the window
   resolution (default folder `renders` in the project).

Structure names can be Allen CCF acronyms (case doesn't matter) or these Paxinos
abbreviations: IRt, PCRt, 12N, Mo5/5N, Pe5/peri5, Acs5/acc5, 7N/fmn. Separate them with
spaces, commas or slashes, e.g. `irt/pcrt xii mo5 peri5 acc5 fmn`. Add `:colour` to
give a structure its own colour instead of the atlas one, e.g. `irt:red pcrt:#3080ff
xii:gold mo5`; colours are names (`red`, `steelblue`, `gold`, ...) or hex codes, and
*Colour...* picks one for the last structure in the list. The dialog shows what each
name resolves to (for example `peri5 → P5 (Peritrigeminal zone)`) with a swatch of its
colour, and suggests matches for names it doesn't know.

## Command-line export and 3D view

Both are also available without opening the GUI:

```powershell
slicereg export "C:\path\to\slide_project"
slicereg render "C:\path\to\slide_project"
```

`export` (or *Export all cells (CSV)* in the slide view) writes these files into the
project folder:
- `cells_all.csv` has every cell with its atlas coordinates and region.
- `region_counts.csv` has counts per slice, cell type, region and hemisphere.
- `top_regions.csv` ranks the regions by total cell count. Each row has the cell count,
  its percentage of all cells, the left and right hemisphere counts, and with several
  cell types a `cells_<type>` column for each. Cells outside the atlas or in `root`
  aren't ranked but still count towards the percentages. The 3D view's top regions use
  the same ranking.
- `top_regions.png` is a bar plot of the cell count per region, in the same order.

`render` options:

| Option | Meaning |
| --- | --- |
| `--structures NAME ...` | Structures to show, optionally coloured, e.g. `--structures irt:red pcrt:#3080ff xii mo5` |
| `--regions N` | Also show the N regions with the most cells (default 5, 0 for none) |
| `--alpha A` | Structure opacity, 0-1 (default 0.3) |
| `--hemisphere left/right/both` | Which side of the structures to draw |
| `--no-slices` | Hide section images |
| `--no-cells` / `--cell-types T ...` | Hide all cells, or show only these cell types |
| `--no-brain` | Hide the transparent whole brain |
| `--brain-alpha A` | Whole-brain opacity, 0-1 (default 0.3) |
| `--style S` | `plastic` (default), `shiny`, `glossy`, `metallic` or `cartoon` |
| `--view V` | Starting view: `three_quarter` (default), `front`, `back`, `left`, `right`, `top`, `bottom`, `back_low`, `back_low_zoom` |
| `--axes` | Show axes / scale bars |
| `--no-inset` | Hide the small orientation brain in the corner |
| `--cell-size UM` | Radius of the cell spheres in microns (default 25) |
| `--no-title` | Hide the title at the top |
| `--screenshot out.png` | Save an image (2x resolution) instead of opening a window |

For example, only the orofacial brainstem nuclei and cells, without sections:

```powershell
slicereg render "C:\path\to\slide_project" --structures irt/pcrt xii mo5 peri5 acc5 fmn --regions 0 --no-slices
```

## Project folder layout

```
slide_project/
  project.json          # atlas, section boxes, settings, display preferences
  thumbnail.tif/.png    # downsampled slide overview
  slices/slice_01/
    image.tif           # full-resolution crop
    preview.tif         # downsampled crop used for display
    mask.png            # tissue mask
    alignment.json      # atlas alignment
    cells.csv           # counted cells
  cells_all.csv         # after export
  region_counts.csv     # after export
  top_regions.csv       # after export
  top_regions.png       # after export
```

## Tests

```powershell
pytest
```
