#!/usr/bin/env python
"""Interactive MEG sensor layout visualisation for the first subject/session.

This script loads a MEG recording from the local BIDS dataset, coregisters the
sensor locations to a FreeSurfer subject (``fsaverage`` by default) and opens a
3D PyVista scene where the brain surface and MEG sensors are shown together.
Left-click sensors to toggle their selection while rotating/zooming freely.
When the window is closed the selected channel names are printed to stdout so
they can be copied into further scripts or configs.
"""
from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import mne
import numpy as np
import pyvista as pv
from mne.channels import DigMontage, make_dig_montage
from mne.io.constants import FIFF
from mne.coreg import fit_matched_points
from mne.io.kit import read_mrk
from mne.io.kit.coreg import _set_dig_kit
from mne.transforms import (
    als_ras_trans,
    apply_trans,
    get_ras_to_neuromag_trans,
    Transform,
    invert_transform,
)
from mne_bids import BIDSPath, read_raw_bids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an interactive 3D layout of MEG sensors."
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=Path(__file__).parent / "bids_anonym",
        help="Root of the BIDS dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--subject",
        default="01",
        help="BIDS subject (default: %(default)s)",
    )
    parser.add_argument(
        "--session",
        default="0",
        help="BIDS session (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        default="0",
        help="BIDS task/run identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Optional run label.",
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=None,
        help="FreeSurfer SUBJECTS_DIR (default: $SUBJECTS_DIR or fsaverage download).",
    )
    parser.add_argument(
        "--surface-subject",
        default="fsaverage",
        help="FreeSurfer subject providing the brain surface (default: %(default)s)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.2,
        help="Brain surface opacity (default: %(default)s)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=14.0,
        help="MEG sensor marker size (default: %(default)s)",
    )
    parser.add_argument(
        "--sensors-only",
        action="store_true",
        help="Skip MRI surface rendering and only show MEG sensor positions.",
    )
    parser.add_argument(
        "--replot-selected",
        type=Path,
        default=None,
        help=(
            "Path or filename of a previously saved selection to preload and "
            "highlight."
        ),
    )
    return parser.parse_args()


def _resolve_subjects_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    env_dir = os.getenv("SUBJECTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    fsaverage_path = Path(mne.datasets.fetch_fsaverage(verbose=True)).expanduser()
    return fsaverage_path.parent


def _load_raw(bids_root: Path, subject: str, session: str, task: str, run: str | None):
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        suffix="meg",
        extension=".con",
    )
    print(f"Loading MEG data from {bids_path.fpath}")
    raw = read_raw_bids(bids_path=bids_path, verbose="warning")
    return raw


def _meg_session_dir(bids_root: Path, subject: str, session: str) -> Path:
    return bids_root / f"sub-{subject}" / f"ses-{session}" / "meg"


def _selection_dir() -> Path:
    base = Path(__file__).parent / "derivatives" / "channels_selection"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _split_by_hemisphere(
    selected_names: Sequence[str],
    name_to_pos: dict[str, np.ndarray],
    midline_tol: float = 0.005,
) -> tuple[list[str], list[str], list[str]]:
    left, right, central = [], [], []
    for name in selected_names:
        pos = name_to_pos.get(name)
        if pos is None:
            continue
        x_coord = pos[0]
        if abs(x_coord) <= midline_tol:
            central.append(name)
        elif x_coord > 0:
            right.append(name)
        else:
            left.append(name)
    return left, right, central


def _save_selection_file(
    args: argparse.Namespace,
    selected_names: Sequence[str],
    name_to_pos: dict[str, np.ndarray],
) -> Optional[Path]:
    if not selected_names:
        return None
    selection_dir = _selection_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = args.run if args.run is not None else "none"
    filename = (
        f"selection_sub-{args.subject}_ses-{args.session}_task-{args.task}_"
        f"run-{run_label}_{timestamp}.txt"
    )
    left, right, central = _split_by_hemisphere(selected_names, name_to_pos)
    path = selection_dir / filename
    with path.open("w", encoding="utf-8") as fid:
        fid.write("# MEG sensor selection generated by visualisation_layout.py\n")
        fid.write(f"subject={args.subject}\n")
        fid.write(f"session={args.session}\n")
        fid.write(f"task={args.task}\n")
        fid.write(f"run={run_label}\n")
        fid.write(f"timestamp={timestamp}\n\n")
        fid.write("[Left]\n")
        for name in left:
            fid.write(f"{name}\n")
        fid.write("\n[Right]\n")
        for name in right:
            fid.write(f"{name}\n")
        fid.write("\n[Central]\n")
        for name in central:
            fid.write(f"{name}\n")
    return path


def _load_preselected(path: Path | None) -> list[str]:
    if path is None:
        return []
    candidates = [path]
    if not path.is_absolute():
        candidates.append(_selection_dir() / path)
    for candidate in candidates:
        if candidate.exists():
            selected: list[str] = []
            current = None
            with candidate.open("r", encoding="utf-8") as fid:
                for line in fid:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    key = stripped.lower()
                    if key in ("[left]", "left:", "left hemisphere:"):
                        current = "left"
                        continue
                    if key in ("[right]", "right:", "right hemisphere:"):
                        current = "right"
                        continue
                    if key in ("[central]", "central:", "midline:", "midline"):
                        current = "central"
                        continue
                    if current in {"left", "right", "central"}:
                        selected.append(stripped)
            print(f"Loaded {len(selected)} pre-selected sensors from {candidate}.")
            return selected
    raise FileNotFoundError(
        f"Could not locate selection file '{path}'. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def _read_fastscan_pos(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    points = []
    with path.open("r", encoding="utf-8") as fid:
        for line in fid:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            try:
                coords = [float(val) for val in stripped.split()]
            except ValueError:
                continue
            if len(coords) >= 3:
                points.append(coords[:3])
    if not points:
        return None
    pts = np.asarray(points, dtype=float) * 1e-3  # FastSCAN exports mm
    return pts


def _kit_to_head(points: np.ndarray, nmtrans) -> np.ndarray:
    if points is None:
        return None
    pts = np.asarray(points, dtype=float)
    orig_shape = pts.shape
    pts = pts.reshape(-1, 3)
    pts = apply_trans(als_ras_trans, pts)
    pts = apply_trans(nmtrans, pts)
    return pts.reshape(orig_shape)


def _load_digitization_montage(
    bids_root: Path, subject: str, session: str, task: str
) -> Tuple[Optional[DigMontage], Optional[Transform]]:
    meg_dir = _meg_session_dir(bids_root, subject, session)
    elp_path = meg_dir / f"sub-{subject}_ses-{session}_acq-ELP_headshape.pos"
    hsp_path = meg_dir / f"sub-{subject}_ses-{session}_acq-HSP_headshape.pos"
    mrk_path = meg_dir / f"sub-{subject}_ses-{session}_task-{task}_markers.mrk"

    # Preferred path: use the KIT digitizer files directly.
    if elp_path.exists() and hsp_path.exists() and mrk_path.exists():
        elp = _read_fastscan_pos(elp_path)
        hsp = _read_fastscan_pos(hsp_path)
        if elp is not None and hsp is not None:
            try:
                mrk_device = read_mrk(mrk_path)
                dig_points, dev_head_t, _ = _set_dig_kit(
                    mrk=mrk_device, elp=elp, hsp=hsp, eeg=OrderedDict()
                )
                print(
                    "Loaded digitization using KIT FastSCAN files "
                    f"({elp_path.name}, {hsp_path.name}, {mrk_path.name})."
                )
                return DigMontage(dig=dig_points), dev_head_t
            except Exception as exc:
                print(
                    f"KIT digitizer parsing failed ({exc}); falling back to coordsystem."
                )

    coords_path = meg_dir / f"sub-{subject}_ses-{session}_coordsystem.json"
    if not coords_path.exists():
        print(f"No coordsystem file found at {coords_path}, skipping digitization.")
        return None, None
    with coords_path.open("r", encoding="utf-8") as fid:
        coords = json.load(fid)
    anat = coords.get("AnatomicalLandmarkCoordinates")
    if not anat:
        print("coordsystem file lacks anatomical landmark coordinates; skipping dig.")
        return None, None
    try:
        lpa_kit = np.asarray(anat["LPA"], dtype=float)
        nasion_kit = np.asarray(anat["NAS"], dtype=float)
        rpa_kit = np.asarray(anat["RPA"], dtype=float)
    except KeyError as exc:
        print(f"coordsystem is missing {exc.args[0]} entry; cannot build montage.")
        return None, None

    fid_ras = apply_trans(
        als_ras_trans, np.vstack([nasion_kit, lpa_kit, rpa_kit])
    )
    nasion_ras, lpa_ras, rpa_ras = fid_ras
    nmtrans = get_ras_to_neuromag_trans(nasion_ras, lpa_ras, rpa_ras)
    nasion = _kit_to_head(nasion_kit, nmtrans)
    lpa = _kit_to_head(lpa_kit, nmtrans)
    rpa = _kit_to_head(rpa_kit, nmtrans)

    hsp = _read_fastscan_pos(hsp_path)
    if hsp is not None:
        hsp = _kit_to_head(hsp, nmtrans)
        print(f"Loaded {len(hsp)} headshape points from {hsp_path.name}.")

    hpi_points = None
    head_coils = coords.get("HeadCoilCoordinates") or {}
    coil_entries = [
        np.asarray(val, dtype=float)
        for key, val in sorted(head_coils.items())
        if key.upper() not in {"NAS", "LPA", "RPA"}
    ]
    if coil_entries:
        hpi_points = _kit_to_head(np.vstack(coil_entries), nmtrans)
        print(f"Using {len(coil_entries)} head-coil coordinates from {coords_path.name}.")

    mrk_device = None
    if mrk_path.exists():
        try:
            mrk_device = read_mrk(mrk_path)
            print(f"Read {len(mrk_device)} marker coils from {mrk_path.name}.")
        except Exception as exc:  # pragma: no cover - informative warning
            print(f"Failed to read markers from {mrk_path.name}: {exc}")

    montage = make_dig_montage(
        lpa=lpa,
        nasion=nasion,
        rpa=rpa,
        hsp=hsp,
        hpi=hpi_points,
        coord_frame="head",
    )
    dev_head_t = None
    if mrk_device is not None and hpi_points is not None:
        if len(mrk_device) == len(hpi_points):
            trans = fit_matched_points(
                tgt_pts=hpi_points, src_pts=mrk_device, out="trans"
            )
            dev_head_t = Transform("meg", "head", trans)
            print("Derived device→head transform from marker coils.")
        else:
            print(
                f"Marker ({len(mrk_device)}) and head-coil ({len(hpi_points)}) counts "
                "differ; cannot compute device→head transform."
            )
    else:
        print("Insufficient marker/head-coil data for device→head transform.")

    return montage, dev_head_t


def _brain_meshes(
    subject: str,
    subjects_dir: Path,
    trans: mne.transforms.Transform,
) -> Iterable[pv.PolyData]:
    """Return PyVista meshes (head coords) for both hemispheres."""
    mri_to_head = invert_transform(trans)
    for hemi in ("lh", "rh"):
        surf_path = Path(subjects_dir) / subject / "surf" / f"{hemi}.pial"
        rr, tris = mne.read_surface(surf_path)
        rr = apply_trans(mri_to_head, rr * 1e-3)  # FreeSurfer surfaces are in mm
        centroid = rr.mean(axis=0, keepdims=True)
        rr = centroid + 1.2 * (rr - centroid)
        faces = np.hstack([np.full((tris.shape[0], 1), 3), tris]).astype(np.int64)
        mesh = pv.PolyData(rr, faces)
        yield mesh


def _sensor_positions(info: mne.Info) -> tuple[np.ndarray, list[str]]:
    picks = mne.pick_types(info, meg=True, ref_meg=False)
    positions = []
    names = []
    dev_head = info.get("dev_head_t")
    for idx in picks:
        loc = info["chs"][idx]["loc"][:3]
        if not np.any(loc):
            continue
        if dev_head is not None and info["chs"][idx]["coord_frame"] == FIFF.FIFFV_COORD_DEVICE:
            loc = apply_trans(dev_head, loc)
        positions.append(loc.copy())
        names.append(info["ch_names"][idx])
    if not positions:
        raise RuntimeError("No MEG sensors with valid positions were found in raw info.")
    positions = np.asarray(positions)
    return positions, names


def _setup_plot(
    brain_meshes: Iterable[pv.PolyData],
    sensor_positions: np.ndarray,
    sensor_names: list[str],
    opacity: float,
    point_size: float,
    *,
    show_brain: bool = True,
    initial_selection: Optional[Sequence[str]] = None,
) -> tuple[pv.Plotter, pv.PolyData, OrderedDict[str, None]]:
    pv.global_theme.smooth_shading = True
    plotter = pv.Plotter()
    if show_brain:
        for mesh in brain_meshes:
            plotter.add_mesh(
                mesh,
                color="#bbbbbb",
                opacity=opacity,
                pickable=False,
                smooth_shading=True,
                show_scalar_bar=False,
            )

    sensor_mesh = pv.PolyData(sensor_positions)
    sensor_mesh["selected"] = np.zeros(len(sensor_names))
    name_to_index = {name: idx for idx, name in enumerate(sensor_names)}
    plotter.add_mesh(
        sensor_mesh,
        scalars="selected",
        cmap=["#1f77b4", "#d62728"],
        clim=[0, 1],
        point_size=point_size,
        render_points_as_spheres=True,
        show_scalar_bar=False,
        name="meg_sensors",
    )
    plotter.add_axes()
    plotter.show_grid(color="#aaaaaa")
    plotter.add_text(
        "Right click sensors to toggle selection.\nClose window to print/save choices.",
        font_size=10,
    )
    selected = OrderedDict()
    if initial_selection:
        scalars = sensor_mesh["selected"]
        for name in initial_selection:
            idx = name_to_index.get(name)
            if idx is None:
                continue
            scalars[idx] = 1
            selected[name] = None
        sensor_mesh["selected"] = scalars

    def _hemi_label(idx: int, tol: float = 0.005) -> str:
        x_coord = sensor_positions[idx, 0]
        if abs(x_coord) <= tol:
            return "central"
        return "right" if x_coord > 0 else "left"

    def _on_pick(point):
        if point is None:
            return
        distances = np.linalg.norm(sensor_positions - point, axis=1)
        idx = int(np.argmin(distances))
        name = sensor_names[idx]
        hemi = _hemi_label(idx)
        scalars = sensor_mesh["selected"]
        if scalars[idx] > 0:
            scalars[idx] = 0
            selected.pop(name, None)
            print(f"Removed {name} ({hemi})")
        else:
            scalars[idx] = 1
            selected[name] = None
            print(f"Added {name} ({hemi})")
        sensor_mesh["selected"] = scalars
        plotter.render()

    plotter.enable_point_picking(
        callback=_on_pick,
        show_message=True,
        point_size=10,
        show_point=False,
        pickable_window=True,
    )
    return plotter, sensor_mesh, selected


def main() -> None:
    args = _parse_args()
    bids_root = args.bids_root.resolve()
    raw = _load_raw(bids_root, args.subject, args.session, args.task, args.run)
    montage, dev_head_t = _load_digitization_montage(
        bids_root, args.subject, args.session, args.task
    )
    if montage is not None:
        print("Applying digitization montage to raw info.")
        raw.set_montage(montage, on_missing="ignore")
    if dev_head_t is not None:
        raw.info["dev_head_t"] = dev_head_t
    preselected = _load_preselected(args.replot_selected)
    sensor_positions, sensor_names = _sensor_positions(raw.info)
    name_to_pos = {name: pos for name, pos in zip(sensor_names, sensor_positions)}
    initial_selection = []
    if preselected:
        for name in preselected:
            if name in name_to_pos:
                initial_selection.append(name)
            else:
                print(f"Warning: selection entry '{name}' not found in this run.")

    brain_meshes: tuple[pv.PolyData, ...] = tuple()
    if not args.sensors_only:
        subjects_dir = _resolve_subjects_dir(args.subjects_dir)
        trans = None
        if montage is not None:
            try:
                dig_positions = montage.get_positions()
                coreg = mne.coreg.Coregistration(
                    raw.info, subject=args.surface_subject, subjects_dir=subjects_dir
                )
                coreg.fit_fiducials(verbose=True)
                if dig_positions.get("hsp") is not None:
                    coreg.fit_icp(verbose=True)
                else:
                    print("No headshape points available; skipping ICP refinement.")
                trans = coreg.trans
                print("Computed head↔MRI transform using digitized fiducials/headshape.")
            except Exception as exc:
                print(f"Digitizer-based coregistration failed: {exc}")

        if trans is None:
            trans = mne.coreg.estimate_head_mri_t(
                args.surface_subject, subjects_dir=subjects_dir
            )
            print("Using fsaverage fiducials for an approximate head↔MRI transform.")
        brain_meshes = tuple(
            _brain_meshes(args.surface_subject, subjects_dir, trans)
        )  # freeze for reuse

    plotter, _, selected = _setup_plot(
        brain_meshes,
        sensor_positions,
        sensor_names,
        args.opacity,
        args.point_size,
        show_brain=not args.sensors_only,
        initial_selection=initial_selection,
    )
    plotter.show(title="MEG sensor layout", auto_close=True)
    selected_names = list(selected.keys())
    if selected_names:
        print("\nSelected sensors:")
        for name in selected_names:
            print(name)
    else:
        print("\nNo sensors were selected.")
    output_path = _save_selection_file(args, selected_names, name_to_pos)
    if output_path is not None:
        print(f"\nSaved selection to {output_path}.")


if __name__ == "__main__":
    main()
