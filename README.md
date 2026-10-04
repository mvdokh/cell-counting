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
| `pixel_um` | `null` | Pixel size in microns; `null` reads it from the file |
| `atlas`, `section_spacing_um` | `allen_mouse_25um`, 100 | Used when the project is created |

The slide view then shows every section at true size in its column/row, cropped to
its tissue and with each section's brightness stretched separately so sections
imaged with different settings look alike (the slice view itself shows the raw
data). Double-click
one to align and count it exactly as above. Z-stacks are max-projected, and mosaics
that were saved as separate tiles are stitched from the tile positions in the file.
Click *Rescan folder* after editing the config or adding images; existing sections
keep their alignments and cells.

Options for `open`:

| Option | Meaning |
| --- | --- |
| `--project DIR` | Put the project folder somewhere else |
| `--atlas NAME` | BrainGlobe atlas for a new project, e.g. `allen_mouse_10um` |
| `--spacing UM` | Section spacing in microns, used to guess the next slice's AP position |

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
8. Back in the slide view, click **Export all cells (CSV)** or
   **3D view in brainrender**. The 3D view button opens an options dialog: type the
   structures to show (see below), choose how many top regions by cell count to add,
   the structure opacity and hemisphere, and whether to show section images, cells
   (per cell type) and the whole-brain outline. The last choices are remembered.

Structure names can be Allen CCF acronyms (case doesn't matter) or these Paxinos
abbreviations: IRt, PCRt, 12N, Mo5/5N, Pe5/peri5, Acs5/acc5, 7N/fmn. Separate them with
spaces, commas or slashes, e.g. `irt/pcrt xii mo5 peri5 acc5 fmn`. The dialog shows
what each name resolves to (for example `peri5 → P5 (Peritrigeminal zone)`) and
suggests matches for names it doesn't know.

## Command-line export and 3D view

Both are also available without opening the GUI:

```powershell
slicereg export "C:\path\to\slide_project"
slicereg render "C:\path\to\slide_project"
```

`export` writes `cells_all.csv` (every cell with atlas coordinates and region) and
`region_counts.csv` (counts per slice, cell type, region and hemisphere) into the
project folder.

`render` options:

| Option | Meaning |
| --- | --- |
| `--structures NAME ...` | Structures to show, e.g. `--structures irt/pcrt xii mo5 peri5 acc5 fmn` |
| `--regions N` | Also show the N regions with the most cells (default 5, 0 for none) |
| `--alpha A` | Structure opacity, 0-1 (default 0.3) |
| `--hemisphere left/right/both` | Which side of the structures to draw |
| `--no-slices` | Hide section images |
| `--no-cells` / `--cell-types T ...` | Hide all cells, or show only these cell types |
| `--no-brain` | Hide the transparent whole-brain outline |
| `--screenshot out.png` | Save an image instead of opening a window |

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
```

## Tests

```powershell
pytest
```
