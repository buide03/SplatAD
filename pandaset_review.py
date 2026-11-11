"""Review and patch native PandaSet cuboids without modifying the source sequence."""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from pandaset_data import CAMERAS, validate_sequence


XYZ = ["position.x", "position.y", "position.z"]
DIMS = ["dimensions.x", "dimensions.y", "dimensions.z"]
GEOMETRY = XYZ + DIMS + ["yaw"]
CASES = ("walking", "cycling", "occlusion")
EXTRA_CASES = ("cycling_occlusion", "walking_pair")
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cuboid_file(root, frame):
    if type(frame) is not int or not 0 <= frame < 80:
        raise ValueError(f"Invalid frame {frame!r}")
    return Path(root) / "annotations" / "cuboids" / f"{frame:02d}.pkl.gz"


def camera_data(root, camera):
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera: {camera}")
    folder = Path(root) / "camera" / f"{camera}_camera"
    return folder, load_json(folder / "intrinsics.json"), load_json(folder / "poses.json"), load_json(folder / "timestamps.json")


def project(points, intrinsics, pose):
    heading = pose["heading"]
    rotation = Rotation.from_quat([heading[k] for k in "xyzw"])
    translation = np.array([pose["position"][k] for k in "xyz"])
    camera_xyz = rotation.inv().apply(points - translation)
    z = camera_xyz[:, 2]
    xy = camera_xyz[:, :2] / np.maximum(z[:, None], 1e-6)
    return xy * [intrinsics["fx"], intrinsics["fy"]] + [intrinsics["cx"], intrinsics["cy"]], z


def box_corners(row):
    local = SIGNS * row[DIMS].to_numpy(dtype=float) / 2
    return Rotation.from_euler("z", float(row["yaw"])).apply(local) + row[XYZ].to_numpy(dtype=float)


def case_track_ids(case):
    return (case["uuid"], case["secondary_uuid"]) if "secondary_uuid" in case else (case["uuid"],)


def overlay(root, camera, frame, track_uuid=None, show_lidar=True):
    folder, intrinsics, poses, times = camera_data(root, camera)
    img = Image.open(folder / f"{frame:02d}.jpg").convert("RGB")
    draw = ImageDraw.Draw(img)
    if show_lidar:
        lidar_root = Path(root) / "lidar"
        lidar_times = np.array(load_json(lidar_root / "timestamps.json"))
        scan = int(np.abs(lidar_times - times[frame]).argmin())
        points = pd.read_pickle(lidar_root / f"{scan:02d}.pkl.gz")
        points = points.loc[points["d"] == 0, ["x", "y", "z"]].to_numpy()[::5]
        xy, z = project(points, intrinsics, poses[frame])
        visible = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < img.width) & (xy[:, 1] >= 0) & (xy[:, 1] < img.height)
        draw.point([tuple(p) for p in xy[visible]], fill="#5ce9fe")
    boxes = pd.read_pickle(cuboid_file(root, frame))
    boxes = boxes.loc[~boxes.stationary & (boxes["cuboids.sensor_id"] != 1)]
    colors = {}
    if track_uuid is not None:
        ids = (track_uuid,) if isinstance(track_uuid, str) else track_uuid
        boxes = boxes.loc[boxes.uuid.isin(ids)]
        colors = dict(zip(ids, ("#f5cb52", "#a8e173")))
    for _, box in boxes.iterrows():
        xy, z = project(box_corners(box), intrinsics, poses[frame])
        if (z <= 0).any() or (xy.max(axis=0) < 0).any() or (xy.min(axis=0) >= img.size).any():
            continue
        for a, b in EDGES:
            draw.line([tuple(xy[a]), tuple(xy[b])], fill=colors.get(box.uuid, "#f5cb52"), width=3)
        anchor = tuple(np.clip(xy.min(axis=0), (0, 0), (img.width - 1, img.height - 1)))
        draw.text(anchor, f"{box.label} {box.uuid[:8]}", fill="#ffffff", stroke_width=2, stroke_fill="#000000")
    return img


def write_overlay_clip(root, case, output, fps=5):
    """Encode the reviewed track as an MP4. Frame IDs stay visible over gaps."""
    import subprocess

    with tempfile.TemporaryDirectory() as temp:
        for index, frame in enumerate(range(case["start_frame"], case["end_frame"] + 1)):
            img = overlay(root, case["camera"], frame, case_track_ids(case), show_lidar=False)
            ImageDraw.Draw(img).text((16, 16), f"Frame {frame:02d}", fill="white", stroke_width=2, stroke_fill="black")
            img.thumbnail((960, 540))
            img.save(Path(temp) / f"{index:04d}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i",
                        str(Path(temp) / "%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)], check=True)


def track_table(root, uuid):
    rows = []
    times = load_json(Path(root) / "lidar" / "timestamps.json")
    for frame in range(80):
        boxes = pd.read_pickle(cuboid_file(root, frame))
        match = boxes.loc[(boxes.uuid == uuid) & (boxes["cuboids.sensor_id"] != 1)]
        if len(match) > 1:
            raise ValueError(f"Duplicate UUID {uuid} at frame {frame}")
        if len(match):
            rows.append({"frame": frame, "timestamp": times[frame], **match.iloc[0][["label", "stationary", *GEOMETRY]].to_dict()})
    table = pd.DataFrame(rows)
    if len(table):
        table["gap_frames"] = table.frame.diff().fillna(1).astype(int) - 1
        table["step_m"] = np.linalg.norm(table[XYZ].diff().to_numpy(), axis=1)
    return table


def validate_cases(cases, required=True):
    names = [case["case"] for case in cases]
    if len(names) != len(set(names)) or set(names) - set(CASES + EXTRA_CASES) or (required and not set(CASES).issubset(names)):
        raise ValueError("Use unique supported cases and include walking, cycling and occlusion")
    for case in cases:
        if (not isinstance(case["sequence"], str) or len(case["sequence"]) != 3 or not case["sequence"].isdigit()
                or case["camera"] not in CAMERAS or not str(case["uuid"]).strip()
                or not 0 <= case["start_frame"] < case["end_frame"] < 80):
            raise ValueError(f"Invalid case: {case}")
        frames = case["keyframes"]
        ids = case_track_ids(case)
        if (len(ids) != (2 if case["case"] == "walking_pair" else 1)
                or len(set(ids)) != len(ids) or any(not isinstance(uuid, str) or not uuid.strip() for uuid in ids)):
            raise ValueError(f"Invalid track IDs for {case['case']}")
        occluded = case["case"] in {"occlusion", "cycling_occlusion"}
        if set(frames) != ({"before", "during", "after"} if occluded else {"pose_1", "pose_2", "pose_3"}):
            raise ValueError(f"Set three named keyframes for {case['case']}")
        if list(frames.values()) != sorted(set(frames.values())) or any(
            type(frame) is not int or not case["start_frame"] <= frame <= case["end_frame"] for frame in frames.values()
        ):
            raise ValueError(f"Choose three distinct increasing frames in the case window: {case}")


def apply_edits(original, reviewed, edits, journal):
    """Apply reviewed changes to a fresh copy. A failed edit leaves the previous copy intact."""
    original, reviewed = Path(original), Path(reviewed)
    if original.resolve() == reviewed.resolve():
        raise ValueError("Reviewed sequence must be separate from original")
    reviewed.parent.mkdir(parents=True, exist_ok=True)
    audit = []
    with tempfile.TemporaryDirectory(dir=reviewed.parent) as temporary:
        stage = Path(temporary) / reviewed.name
        shutil.copytree(original, stage)
        for edit in edits:
            reason = edit["reason"].strip()
            if not reason:
                raise ValueError("Each edit needs a reason")
            path = cuboid_file(stage, edit["frame"])
            boxes = pd.read_pickle(path)
            matches = boxes.index[(boxes.uuid == edit["uuid"]) & (boxes["cuboids.sensor_id"] != 1)]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous UUID: {edit}")
            before = boxes.loc[matches].to_json(orient="records")
            action = edit["action"]
            if action == "update" and len(matches) == 1:
                values = edit["values"]
                if not values or set(values) - set(GEOMETRY + ["uuid", "label", "stationary"]):
                    raise ValueError(f"Unsupported field update: {values}")
                for field, value in values.items():
                    boxes.loc[matches[0], field] = value
                index = matches[0]
            elif action == "add" and not len(matches):
                values = edit["values"]
                if set(values) - set(boxes) or not set(GEOMETRY + ["label"]).issubset(values) or "uuid" in values:
                    raise ValueError("New boxes need native geometry and label; set UUID on the edit")
                row = {key: None for key in boxes.columns}
                row.update(camera_used=-1, **{"cuboids.sensor_id": -1, "cuboids.sibling_id": "-"})
                row.update(values)
                row.update(uuid=edit["uuid"], stationary=False)
                boxes = pd.concat([boxes, pd.DataFrame([row])], ignore_index=True).astype(boxes.dtypes.to_dict())
                index = boxes.index[-1]
            else:
                raise ValueError(f"Action and UUID do not match existing data: {edit}")
            candidate = boxes.loc[index]
            if (not np.isfinite(candidate[GEOMETRY].to_numpy(dtype=float)).all()
                    or (candidate[DIMS].to_numpy(dtype=float) <= 0).any()
                    or not isinstance(candidate.uuid, str) or not candidate.uuid.strip()
                    or not isinstance(candidate.label, str) or not candidate.label.strip()):
                raise ValueError(f"Invalid box: {edit}")
            native = boxes.loc[boxes["cuboids.sensor_id"] != 1]
            if (native.uuid == candidate.uuid).sum() != 1:
                raise ValueError(f"Duplicate target UUID: {edit}")
            boxes.to_pickle(path)
            audit.append({**edit, "before": json.loads(before), "after": json.loads(boxes.loc[[index]].to_json(orient="records"))})
        validate_sequence(stage)
        previous = Path(temporary) / "previous"
        if reviewed.exists():
            reviewed.replace(previous)
        try:
            stage.replace(reviewed)
        except OSError:
            if previous.exists():
                previous.replace(reviewed)
            raise
    write_json(journal, audit)


def pack_reviewed(root, destination):
    root, destination = Path(root), Path(destination)
    validate_sequence(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    try:
        with zipfile.ZipFile(partial, "w") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, f"pandaset/{root.name}/{path.relative_to(root).as_posix()}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
