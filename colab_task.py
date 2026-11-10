"""Run the operations that need Nerfstudio's Python environment."""

import argparse

from colab_pipeline import audit_parser, export_lidar
from pandaset_review import load_json


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("task", choices=("audit", "lidar"))
    cli.add_argument("spec")
    cli.add_argument("config", nargs="?")
    args = cli.parse_args()
    spec = load_json(args.spec)
    if args.task == "audit":
        audit_parser(spec)
    elif args.config is None:
        cli.error("lidar requires a config.yml")
    else:
        export_lidar(spec, args.config)
