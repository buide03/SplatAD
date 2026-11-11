"""Orchestrate the upstream Nerfstudio CLI and package its recorded outputs."""

import json
import math
import os
import pickle
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from pandaset_review import box_corners, camera_data, case_track_ids, cuboid_file, load_json, project, validate_cases, write_json


RUNTIME = Path("/content/splatad-runtime")
PREFIX = RUNTIME / "env"


def runtime_env():
    env = os.environ.copy()
    env["PATH"] = f"{PREFIX / 'bin'}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{PREFIX / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_HOME"] = str(PREFIX)
    env["CC"] = str(PREFIX / "bin/x86_64-conda-linux-gnu-gcc")
    env["CXX"] = str(PREFIX / "bin/x86_64-conda-linux-gnu-g++")
    env["CUDAHOSTCXX"] = env["CXX"]
    env["MAX_JOBS"] = "2"
    return env


def run(*arguments):
    command = list(map(str, arguments))
    print(shlex.join(command), flush=True)
    subprocess.run(command, env=runtime_env(), check=True)


def time_window(root, cameras, start, end):
    # The pinned parser uses front-camera timestamps for Pandar64 scan indexing.
    channels = [camera_data(root, camera)[3] for camera in sorted({"front", *cameras})]
    low = min(times[0] for times in channels)
    high = max(times[-1] for times in channels)
    start_time = min(times[start] for times in channels) - 1e-5
    end_time = max(times[end] for times in channels) + 1e-5
    return max(0., (start_time - low) / (high - low)), min(1., (end_time - low) / (high - low))


def group_runs(cases, data_root, runs_root, smoke):
    validate_cases(cases)
    groups = {}
    for case in cases:
        groups.setdefault(case["sequence"], []).append(case)
    if smoke:
        primary = next(case for case in cases if case["case"] == "walking")
        groups = {primary["sequence"]: [primary]}
    for sequence, selected in groups.items():
        cameras = sorted({case["camera"] for case in selected})
        if smoke:
            midpoint = (selected[0]["start_frame"] + selected[0]["end_frame"]) // 2
            first = min(56, max(0, midpoint - 12))
            window = time_window(Path(data_root) / sequence, cameras, first, first + 23)
        else:
            window = (0., 1.)
        kind = "smoke" if smoke else "main"
        path = Path(runs_root) / f"{sequence}-{kind}"
        yield {"sequence": sequence, "cases": selected, "cameras": cameras, "window": window,
               "root": str(data_root), "path": str(path), "smoke": smoke}


def audit_parser(spec):
    """Read real parser outputs. Attach source UUIDs to temporary audit metadata only."""
    from unittest.mock import patch

    from nerfstudio.data.dataparsers import pandaset_dataparser as panda

    root, sequence = Path(spec["root"]), spec["sequence"]
    start, end = spec["window"]
    config = panda.PandaSetDataParserConfig(data=root, sequence=sequence, cameras=tuple(spec["cameras"]),
                                           include_deformable_actors=True, train_split_fraction=0.5,
                                           dataset_start_fraction=start, dataset_end_fraction=end,
                                           add_missing_points=True)
    native_group = panda._cuboids_to_trajectories

    def identify(cuboids):
        trajectories = native_group(cuboids)
        ids = list(dict.fromkeys(row["uuid"] for row in cuboids if row["sensor_ids"] != 1))
        if len(trajectories) != len(ids):
            raise RuntimeError("Upstream trajectory grouping changed; review the audit")
        for uuid, trajectory in zip(ids, trajectories):
            trajectory["audit_uuid"] = uuid
        return trajectories

    lidar_times = np.array(camera_data(root / sequence, "front")[3])
    recorded_lidar_times = load_json(root / sequence / "lidar/timestamps.json")
    report = {}
    for split in ("train", "test"):
        with patch.object(panda, "_cuboids_to_trajectories", side_effect=identify):
            output = config.setup().get_dataparser_outputs(split=split)
        cameras = [{"camera": file.parent.name.removesuffix("_camera"), "frame": int(file.stem),
                    "timestamp": float(time) + float(output.time_offset)}
                   for file, time in zip(output.image_filenames, output.cameras.times.flatten())]
        lidars = []
        for i, time in enumerate(output.metadata["lidars"].times.flatten()):
            timestamp = float(time) + float(output.time_offset)
            frame = int(np.abs(lidar_times - timestamp).argmin())
            if abs(lidar_times[frame] - timestamp) > 0.05:
                raise ValueError(f"Unmatched lidar timestamp {timestamp}")
            lidars.append({"index": i, "frame": frame, "timestamp": timestamp,
                           "recorded_lidar_timestamp": recorded_lidar_times[frame]})
        targets = {uuid for case in spec["cases"] for uuid in case_track_ids(case)}
        trajectories = {traj["audit_uuid"]: {
            "label": traj["label"], "timestamps": (traj["timestamps"].double() + float(output.time_offset)).tolist(),
            "centers_model_world": traj["poses"][:, :3, 3].tolist(),
        } for traj in output.metadata["trajectories"] if traj["audit_uuid"] in targets}
        if targets - set(trajectories):
            raise ValueError(f"{split}: target tracks absent after parsing: {sorted(targets - set(trajectories))}")
        report[split] = {"images": cameras, "lidar": lidars, "target_trajectories": trajectories}
        print(f"{sequence} {split}: {len(cameras)} images, {len(lidars)} scans, {len(trajectories)} target tracks")
    train_keys = {(row["camera"], row["frame"]) for row in report["train"]["images"]}
    test_keys = {(row["camera"], row["frame"]) for row in report["test"]["images"]}
    if train_keys & test_keys:
        raise ValueError("Train and held-out image split overlap")
    write_json(Path(spec["path"]) / "split.json", report)
    if not spec["smoke"]:
        for case in spec["cases"]:
            missing = [frame for frame in case["keyframes"].values() if (case["camera"], frame) not in test_keys]
            if missing:
                raise ValueError(f"{case['case']} keyframes {missing} are not held out. See split.json and update cases.json.")


def train(spec, resume=False):
    folder = Path(spec["path"])
    config = folder / "splatad" / "run" / "config.yml"
    count = 500 if spec["smoke"] else 10000
    remaining = count
    load = []
    if config.exists():
        checkpoints = sorted(config.parent.glob("nerfstudio_models/step-*.ckpt"))
        finished = int(checkpoints[-1].stem.split("-")[1]) + 1 if checkpoints else 0
        if finished >= count:
            return config
        if not resume or not checkpoints:
            raise FileExistsError(f"Run exists at {config.parent}; enable resume or choose a new runs directory")
        load = ["--load-dir", checkpoints[-1].parent]
        remaining = count - finished
    args = ["ns-train", "splatad", "--output-dir", folder.parent, "--experiment-name", folder.name,
            "--timestamp", "run", "--vis", "tensorboard", "--max-num-iterations", remaining,
            "--steps-per-save", 250 if spec["smoke"] else 1000, "--steps-per-eval-all-images", 0,
            "--pipeline.datamanager.camera-res-scale-factor", 0.5,
            "--pipeline.datamanager.cache-images", "cpu", "--pipeline.datamanager.cache-lidars", "cpu",
            *load, "pandaset-data", "--data", spec["root"], "--sequence", spec["sequence"],
            "--cameras", *spec["cameras"], "--lidars", "Pandar64", "--include-deformable-actors", "True",
            "--train-split-fraction", 0.5, "--dataset-start-fraction", spec["window"][0],
            "--dataset-end-fraction", spec["window"][1]]
    write_json(folder / "train-command.json", list(map(str, args)))
    for name in ("environment.txt", "commits.txt"):
        shutil.copyfile(RUNTIME / name, folder / name)
    run(*args)
    if not config.is_file() or not list(config.parent.glob("nerfstudio_models/step-*.ckpt")):
        raise FileNotFoundError(f"Nerfstudio did not write config and checkpoint under {config.parent}")
    return config


def render(spec, config):
    output = Path(spec["path"]) / "renders"
    run("ns-render", "dataset", "--load-config", config, "--pose-source", "test", "--output-path", output,
        "--image-format", "png", "--rendered-output-names", "gt-rgb", "rgb", "depth", "raw-depth",
        "--calculate-and-save-metrics", "True")
    write_evaluation(Path(spec["path"]), spec["sequence"])
    return output


IMAGE_METRICS = ("psnr", "ssim", "lpips")


def json_score(score):
    score = float(score)
    if score == math.inf:
        return "Infinity"
    if not math.isfinite(score):
        raise ValueError(f"Invalid image metric: {score}")
    return score


def mean_image_metrics(rows):
    return {name: json_score(sum(float(row[name]) for row in rows) / len(rows)) for name in IMAGE_METRICS}


def write_evaluation(run_dir, sequence):
    """Join metrics from the current native render with the recorded held-out frames."""
    rows = load_json(run_dir / "split.json")["test"]["images"]
    if not rows:
        raise ValueError("No held-out frames to evaluate")
    single_camera = len({row["camera"] for row in rows}) == 1
    with (run_dir / "renders/metrics.pkl").open("rb") as source:
        native = pickle.load(source)  # Written by the ns-render command above.
    frames = []
    for row in rows:
        key = f"{row['frame']:02d}"
        if not single_camera:
            key = f"{row['camera']}_camera/{key}"
        if key not in native or not all(name in native[key] for name in IMAGE_METRICS):
            raise ValueError(f"Missing native PSNR/SSIM/LPIPS for held-out frame {key}")
        frames.append({**row, "metrics": {name: json_score(native[key][name]) for name in IMAGE_METRICS}})
    write_json(run_dir / "renders/evaluation.json", {
        "sequence": sequence, "split": "test", "scope": "full RGB image",
        "frame_count": len(frames), "mean": mean_image_metrics([row["metrics"] for row in frames]),
        "frames": frames,
    })


def write_ply(path, points):
    points = np.asarray(points, dtype="<f4").reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with Path(path).open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(points.tobytes())
    return len(points)


def export_lidar(spec, config):
    import torch
    from nerfstudio.cameras.lidars import transform_points
    from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
    from nerfstudio.scripts.render import _disable_datamanager_setup
    from nerfstudio.utils.eval_utils import eval_setup

    folder = Path(spec["path"])
    records = load_json(folder / "split.json")["test"]
    lidar_rows = records["lidar"]
    if spec["smoke"]:
        chosen = [lidar_rows[len(lidar_rows) // 2]]
    else:
        times = {(row["camera"], row["frame"]): row["timestamp"] for row in records["images"]}
        indices = set()
        for case in spec["cases"]:
            for frame in case["keyframes"].values():
                time = times[(case["camera"], frame)]
                indices.add(min(range(len(lidar_rows)), key=lambda i: abs(lidar_rows[i]["timestamp"] - time)))
        chosen = [lidar_rows[i] for i in sorted(indices)]
    output = folder / "renders" / "lidar"
    output.mkdir(parents=True, exist_ok=True)
    trainer, pipeline, checkpoint, step = eval_setup(Path(config), test_mode="inference")
    dm_config = trainer.pipeline.datamanager
    with _disable_datamanager_setup(dm_config._target):
        dm = dm_config.setup(test_mode="test", device=pipeline.device)
    loader = FixedIndicesEvalDataloader(dataset=dm.eval_lidar_dataset, device=pipeline.device, num_workers=0)
    result = []
    with torch.no_grad():
        for row in chosen:
            lidar, batch = loader.get_camera(row["index"])
            prediction, batch = pipeline.model.get_outputs_for_lidar(lidar, batch=batch)
            keep = prediction["points_valid"] & (prediction["ray_drop_prob"].squeeze(-1) < 0.5)
            predicted = transform_points(prediction["points"][keep], lidar.lidar_to_worlds[0])
            measured = batch["lidar"][..., :3]
            returned = batch.get("lidar_pts_did_return")
            if returned is not None:
                measured = measured[returned.reshape(-1).bool()]
            measured = transform_points(measured, lidar.lidar_to_worlds[0])
            counts = {}
            for name, cloud in [("prediction", predicted), ("ground_truth", measured)]:
                counts[name] = write_ply(output / f"{row['frame']:02d}-{name}.ply", cloud.cpu().numpy())
            result.append({**row, "vertices": counts, "coordinates": "Nerfstudio model world (meters)"})
    write_json(output / "scans.json", {"checkpoint": checkpoint.name, "step": step, "scans": result})


def projected_crop(data_root, case, frame):
    import pandas as pd

    root = Path(data_root) / case["sequence"]
    folder, intrinsics, poses, _ = camera_data(root, case["camera"])
    boxes = pd.read_pickle(cuboid_file(root, frame))
    ids = case_track_ids(case)
    boxes = boxes.loc[boxes.uuid.isin(ids) & (boxes["cuboids.sensor_id"] != 1)]
    if len(boxes) != len(ids) or boxes.uuid.nunique() != len(ids):
        return None
    corners = np.concatenate([box_corners(box) for _, box in boxes.iterrows()])
    xy, z = project(corners, intrinsics, poses[frame])
    if np.any(z <= 0):
        return None
    with Image.open(folder / f"{frame:02d}.jpg") as original:
        width, height = original.size
    if case["camera"] == "back":
        height -= 260  # Crop performed by the pinned PandaSet parser.
    low, high = xy.min(axis=0), xy.max(axis=0)
    padding = np.maximum((high - low) * 0.35, 16)
    lo = np.clip((low - padding) / (width, height), 0, 1)
    hi = np.clip((high + padding) / (width, height), 0, 1)
    return [*lo.tolist(), *(hi - lo).tolist()] if np.all(hi > lo) else None


def pack_demo(cases, runs_root, data_root, viewer_source, output, fps=5):
    """Package only matched held-out RGB pairs and their metadata for the static viewer."""
    validate_cases(cases)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(viewer_source) / "index.html", output / "index.html")
    manifest = {"version": 1, "fps": fps, "cases": []}
    for case in cases:
        run_dir = Path(runs_root) / f"{case['sequence']}-main"
        report = load_json(run_dir / "split.json")["test"]["images"]
        evaluation_path = run_dir / "renders/evaluation.json"
        metrics = {(row["camera"], row["frame"]): row["metrics"]
                   for row in load_json(evaluation_path)["frames"]} if evaluation_path.is_file() else {}
        rows = sorted((row for row in report if row["camera"] == case["camera"]
                       and case["start_frame"] <= row["frame"] <= case["end_frame"]), key=lambda row: row["timestamp"])
        keyframes = {frame: role for role, frame in case["keyframes"].items()}
        if set(keyframes) - {row["frame"] for row in rows}:
            raise ValueError(f"Missing held-out keyframe for {case['case']}")
        media = output / "media" / case["case"]
        media.mkdir(parents=True, exist_ok=True)
        entry = {k: case[k] for k in ("case", "sequence", "camera", "uuid", "secondary_uuid") if k in case}
        lidar_manifest = load_json(run_dir / "renders/lidar/scans.json")
        entry.update(checkpoint=lidar_manifest["checkpoint"], frames=[], videos={})
        single_camera = len({row["camera"] for row in report}) == 1
        notes = case.get("observations", {})
        for i, row in enumerate(rows):
            frame = row["frame"]
            rel = Path(f"{frame:02d}.png")
            if not single_camera:
                rel = Path(f"{case['camera']}_camera") / rel
            observation = notes.get(str(frame), {})
            crop = observation.get("crop") or projected_crop(data_root, case, frame)
            if crop is not None:
                if (len(crop) != 4 or min(crop) < 0 or min(crop[2:]) <= 0
                        or crop[0] + crop[2] > 1.000001 or crop[1] + crop[3] > 1.000001):
                    raise ValueError(f"Invalid normalized crop for {case['case']} frame {frame}")
            item = {"frame": frame, "timestamp": row["timestamp"], "role": keyframes.get(frame),
                    "observation": observation.get("note", ""), "crop": crop}
            if metrics:
                item["metrics"] = metrics[(row["camera"], frame)]
            dimensions = []
            for stream, name in (("gt-rgb", "original"), ("rgb", "rendered")):
                source = run_dir / "renders/test" / stream / rel
                dest = media / f"{name}-{i:04d}.png"
                shutil.copyfile(source, dest)
                with Image.open(dest) as image:
                    dimensions.append(image.size)
                item[name] = dest.relative_to(output).as_posix()
            depth_source = run_dir / "renders/test/depth" / rel
            if depth_source.is_file():
                depth_file = media / f"depth-{i:04d}.png"
                shutil.copyfile(depth_source, depth_file)
                with Image.open(depth_file) as image:
                    dimensions.append(image.size)
                item["depth"] = depth_file.relative_to(output).as_posix()
            if len(set(dimensions)) != 1:
                raise ValueError(f"Image dimensions differ for {case['case']} frame {frame}")
            entry["frames"].append(item)
        if metrics:
            entry["metrics"] = mean_image_metrics([item["metrics"] for item in entry["frames"]])
        for name in ("original", "rendered"):
            clip = media / f"{name}.mp4"
            run("ffmpeg", "-loglevel", "error", "-y", "-framerate", fps,
                "-i", media / f"{name}-%04d.png", "-frames:v", len(rows), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", clip)
            entry["videos"][name] = clip.relative_to(output).as_posix()
        manifest["cases"].append(entry)
    write_json(output / "manifest.json", manifest)
    shutil.make_archive(str(output), "zip", output)
