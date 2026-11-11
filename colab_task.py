"""Run the operations that need Nerfstudio's Python environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from colab_pipeline import audit_parser, export_lidar
from pandaset_review import load_json


def open_viewer(config_path, data_root, step):
    """Open the native viewer after copying a trusted Colab run to a Linux GPU host."""
    import yaml

    config_path = Path(config_path).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    manager = config.pipeline.datamanager
    sequence = manager.dataparser.sequence
    if not (data_root / sequence / "annotations/cuboids").is_dir():
        raise FileNotFoundError(f"Expected the reviewed PandaSet sequence at {data_root / sequence}")
    checkpoints = config_path.parent / config.relative_model_dir
    candidates = {int(file.stem[5:]): file for file in checkpoints.glob("step-*.ckpt")
                  if file.stem[5:].isdigit()}
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints}")
    step = max(candidates) if step is None else step
    if step not in candidates:
        raise FileNotFoundError(f"No checkpoint at step {step} in {checkpoints}")

    manager.data = data_root
    manager.dataparser.data = data_root
    config.load_dir = checkpoints
    config.load_step = step
    with tempfile.TemporaryDirectory(prefix="splatad-viewer-") as temporary:
        config.output_dir = Path(temporary)
        viewer_config = Path(temporary) / "config.yml"
        viewer_config.write_text(yaml.dump(config), encoding="utf-8")
        print(f"Opening {candidates[step].name} at http://127.0.0.1:7007", flush=True)
        subprocess.run([sys.executable, "-m", "nerfstudio.scripts.viewer.run_viewer", "--load-config",
                        str(viewer_config), "--viewer.websocket-host", "127.0.0.1",
                        "--viewer.websocket-port", "7007"], check=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("task", choices=("audit", "lidar", "viewer"))
    cli.add_argument("spec")
    cli.add_argument("config", nargs="?")
    cli.add_argument("--data", type=Path, help="PandaSet root for the native viewer")
    cli.add_argument("--step", type=int, help="Checkpoint step for the native viewer")
    args = cli.parse_args()
    if args.task == "viewer":
        if args.data is None or args.config is not None:
            cli.error("viewer requires: viewer <config.yml> --data <PandaSet root> [--step <number>]")
        open_viewer(args.spec, args.data, args.step)
    elif args.task == "audit":
        audit_parser(load_json(args.spec))
    elif args.config is None:
        cli.error("lidar requires a config.yml")
    else:
        export_lidar(load_json(args.spec), args.config)
