"""Functions for photogrammetric coregistration of fNIRS montage."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
import xarray as xr

from cedalion import dataclasses as cdc
from cedalion import io, nirs, typing, units
from cedalion.vis import blocks, anatomy
from cedalion.geometry import registration
from cedalion.geometry.photogrammetry import processors

xr.set_options(display_expand_data=False)

STICKERS_CLASS = (
    tuple[typing.LabeledPoints, xr.DataArray]
    | tuple[
        typing.LabeledPoints,
        xr.DataArray,
        processors.ColoredStickerProcessorDetails,
    ]
)


def find_stickers(
    mesh: cdc.TrimeshSurface, colors: dict[str, Any] | None = None, **kwargs
) -> STICKERS_CLASS:
    """Find stickers inside Mesh.

    This function imports an OBJ file and tries to find stickers in it (by default, yellow circles).

    Args:
        mesh (TrimeshSurface): Mesh in which to look for stickers.
        colors (dict[str, Any] | None, optional): Rage of colors to look for. Defaults to bright yellow.
        **kwargs: Extra arguments passed to `processors.ColoredStickerProcessor`.

    Returns:
        tuple[LabeledPoints, DataArray, ColoredStickerProcessorDetails]: Extracted sticker positions, surface normal vectors, and sticker details.
    """
    if colors is None:
        colors = {"O": ((0.09, 0.22, 0.5, 1))}

    proc = processors.ColoredStickerProcessor(
        colors=colors, sticker_radius=0.8 * units.cm, **kwargs
    )
    return proc.process(mesh, details=True)


def manual_corrections(
    mesh: cdc.TrimeshSurface,
    stickers: xr.DataArray,
    normals: xr.DataArray | None = None,
) -> anatomy.OptodeSelector:
    """Manually correct missing or wrong sticker positions using a GUI.

    Args:
        mesh (TrimeshSurface): Mesh in which to look for stickers.
        stickers (DataArray): Extracted sticker positions.
        normals (DataArray): Surface normal vectors.

    Returns:
        OptodeSelector: Corrected sticker information.
    """
    optode_selector = anatomy.OptodeSelector(mesh, stickers, normals)
    optode_selector.plot()
    optode_selector.enable_picking()

    blocks.plot_surface(optode_selector.plotter, mesh, opacity=1.0)
    optode_selector.plotter.show()

    return optode_selector


def get_scalp_coords(
    stickers: xr.DataArray, normals: xr.DataArray, optode_len: float = 22.6 * units.mm
) -> xr.DataArray:
    """Get scalp coordinates from stickers by subtracting optode lenght fom sticker position.

    Args:
        stickers (DataArray): Extracted sticker positions.
        normals (DataArray): Surface normal vectors.
        optode_len (float, optional): Length of the optodes (how separated the stickers are from the scalp surface). Defaults to 22.6 mm.

    Returns:
        DataArray: Scalp coordinates.
    """
    scalp_coords = stickers.copy()
    mask = stickers.group == "O"
    scalp_coords[mask] = stickers[mask] - optode_len * normals[mask]

    return scalp_coords


def pick_landmarks(mesh: cdc.TrimeshSurface):
    """Pick anatomical landmarks manually in a 3D plot.

    Args:
        mesh (cdc.TrimeshSurface): 3D scan of the montage.
    """
    pvplt = pv.Plotter()
    labels = ["Nz", "Cz", "Lpa", "Rpa"]
    get_landmarks = blocks.plot_surface(pvplt, mesh, opacity=1.0, pick_landmarks=labels)
    pvplt.show()

    return get_landmarks(), labels


def wrap_landmarks(coords: list[np.ndarray], labels: list[str]):
    """Snap anatomical landmark locations to closest point in the Mesh surface in the 3D scan.

    Args:
        coords (list[np.ndarray]): Landmark locations, as returned by `manual_landmarks`. Landmark labels.
    """
    dims = ["label", "digitized"]
    types = [cdc.PointType.LANDMARK] * len(labels)
    groups = ["L"] * len(labels)

    coords_dict = {
        "label": (dims[0], labels),
        "type": (dims[0], types),
        "group": (dims[0], groups),
    }

    landmarks = xr.DataArray(np.vstack(coords), dims=dims, coords=coords_dict)

    return landmarks.pint.quantify("mm")


def transform_montage(
    mesh: cdc.TrimeshSurface,
    montage: xr.DataArray,
    landmarks: xr.DataArray,
    plot: bool = False,
):
    """Snap sensor locations to closest points in digitized surface.

    Args:
        mesh (cdc.TrimeshSurface): Scanned montage.
        montage (xr.DataArray): Original montage to be adjusted.
        landmarks (xr.DataArray): Landmark labels and locations.
        plot (bool, optional): Should the original and adjusted montages be plotted side by side for comparison? Defaults to False.

    """
    trans = registration.register_trans_rot(landmarks, montage)
    idx = (montage.type == cdc.PointType.SOURCE) | (
        montage.type == cdc.PointType.DETECTOR
    )
    filtered_mon = montage.where(idx, drop=True).points.apply_transform(trans)

    if plot:
        pvplt = pv.Plotter()
        blocks.plot_surface(pvplt, mesh, color="w", opacity=0.2)
        blocks.plot_labeled_points(pvplt, filtered_mon)
        pvplt.show()

    return filtered_mon


def adjust_coords(montage: xr.DataArray, landmarks: xr.DataArray, scalp_coords):
    """Iterative adjustment of sensor locations in the scanned surface based on landmark locations.

    Args:
        montage (xr.DataArray): Original montage to be adjusted.
        landmarks (xr.DataArray): Landmark labels and locations.
        scalp_coords (xr.DataArray): Scalp locations.

    Returns:
        tuple (xr.DataArray, xr.DataArray): Scalp coordinates and adjusted landmarks.
    """
    idx = registration.icp_with_full_transform(
        scalp_coords, montage, max_iterations=100
    )

    # extract labels for detected optodes
    label_dict = {}
    for i, label in enumerate(montage.coords["label"].values):
        label_dict[i] = label

    labels = [label_dict[index] for index in idx]

    # write labels to scalp_coords and add landmarks
    scalp_coords = scalp_coords.assign_coords(label=labels)

    return scalp_coords, landmarks


def plot_adjusted_montage(
    x: cdc.Recording, montage: xr.DataArray, montage_adj: xr.DataArray
):
    """Plot original and adjusted montages side by side for comparison.

    Args:
        x (cdc.Recording): Original recording.
        montage (xr.DataArray): Original montage.
        montage_adj (xr.DataArray): Photogrammetry-adjusted montage.
    """
    ch_dist_1 = nirs.channel_distances(x["amp"], montage)
    ch_dist_2 = nirs.channel_distances(x["amp"], montage_adj)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    anatomy.scalp_plot(
        x["amp"].copy(),
        montage,
        ch_dist_1,
        ax=ax[0],
        optode_labels=True,
        cb_label="Channel distance (mm)",
        cmap="plasma",
        vmin=25,
        vmax=42,
    )
    ax[0].set_title("Original montage")

    anatomy.scalp_plot(
        x["amp"].copy(),
        montage_adj,
        ch_dist_2,
        ax=ax[1],
        optode_labels=True,
        cb_label="Channel distance (mm)",
        cmap="plasma",
        vmin=25,
        vmax=42,
    )
    ax[1].set_title("Photogrammetry-adjusted montage")

    fig.tight_layout()
    fig.savefig("img.png")


def coregister_montage(rec: cdc.Recording, obj_file: Path, plot: bool = True, **kwargs):
    """Adjust recoridng's montage using photogrammetric spatial registration routine.

    Args:
        x (xr.DataArray): Original recording.
        obj_file (Path): Path to the OBJ file with scanned montage.
        plot (bool, optional): Should the original and the adjusted montages be plotted side by side for comparison? Defaults to True.
    """
    x = deepcopy(rec)
    mesh = io.read_einstar_obj(str(obj_file))

    stickers, normals, *_ = find_stickers(mesh, **kwargs)

    opt_selector = manual_corrections(mesh, stickers, normals)
    scalp_coords = get_scalp_coords(opt_selector.points, opt_selector.normals)  # ty: ignore[invalid-argument-type]

    landmark_coords, landmark_labels = pick_landmarks(mesh)
    landmarks = wrap_landmarks(landmark_coords, landmark_labels)

    mon = x.geo3d.copy()
    mon = mon.points.rename({"LPA": "Lpa", "RPA": "Rpa"})

    mon_adj = transform_montage(mesh, mon.copy(), landmarks)
    scalp_coords_adj, landmarks_adj = adjust_coords(mon_adj, landmarks, scalp_coords)

    geo3d_adj = processors.geo3d_from_scan(scalp_coords_adj, landmarks_adj)
    ch_select = [ch for ch in geo3d_adj.label.values if ch in x.geo3d.label.values]
    x.geo3d = x.geo3d.sel(label=ch_select + ["Nz", "Cz", "LPA", "RPA"])

    if plot:
        plot_adjusted_montage(x, mon, geo3d_adj)

    return x


if __name__ == "__main__":
    DATA_PATH = Path("data")
    obj_file = DATA_PATH / "scan.obj"
    snirf_file = DATA_PATH / "recording.snirf"

    rec = io.read_snirf(snirf_file, crs="crs")[0]
    rec_adj = coregister_montage(rec, obj_file)

    anatomy.plot_montage3D(rec["amp"], rec.geo3d)
    anatomy.plot_montage3D(rec_adj["amp"], rec_adj.geo3d)
