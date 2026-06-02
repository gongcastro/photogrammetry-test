# photogrammetry-test

An attempt to spatially reigster an fNIRS montage using the photrogrammetry routine in Cedalion v26.5.1. This repo is a derivation of Cedalion's [Photogrammetric Optode Coregistration](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples/head_models/41_photogrammetric_optode_coregistration.html) tutorial, adapted to work on in-house recordings and scans at the NeuroDevCo research group at Institut de Rercerca Sant Joan de Déu.

The can was captured using an Apple [iPhone 12](https://support.apple.com/es-es/111876) in one of the maternity ward rooms, with dim lighting. The models is a doll placed inside one of the cribs for a realistic scenario.

## Setup

0. Install Python (ideally [3.13.12](https://www.python.org/downloads/release/python-31312/))
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to set up your virtual environment and install Python dependencies.
2. Clone or download this repository into your local machine and open a terminal inside it.
3. Install Python dependencies:

```bash
uv sync
```

4. Run the `main.py` script:

```bash
uv run main.py
```

1. First, you will need to **manually select any stickers** that Cedalion's functions have not automatically found, either because of color differences (e.g., shades in scan modify colors) or because of scan quality (some optodes are difficult to scan, specially thos in more posterior locations). Red dots should sit in the middle of the yellow stickers, on top of each optode. Remove red dots or place new ones by right clicking on them. Once happy with the result, close the window.
2. Second, you will need to **indicate some anatomical landmarks**: *Nz*, *Cz*, *LPA*, *RPA*. As before, right click to place the landmarks. Right click on an existing change then label.

> [!TIP]
>Please, set the labels in the indicated order (I noticed some unexpected behaviour when selecting the labels in a different order). If landmark labels are not visible, try zooming in.

3. After some calculations, a plot is generated, comparing the original and the photogrametry-adjusted montages.

> [!WARNING]
>Since the model is a doll is unrealistic head morphology, the result might look a bit weird, but hopefully will illustrate that the transformations have occurred.

4. If you inspect the `main.py` script, you'll see that the `coregister_montage` function (the one wrapping every other function) returns an object `rec_adj`. This is the original recording with the adjusted montage. Further preprocessing steps (e.g., chromophore concentration calculation) should take the new source-detector distances into account.
