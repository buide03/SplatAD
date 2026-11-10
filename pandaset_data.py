"""Download and stage the author's PandaSet ZIP without extracting unrelated scenes."""

import io
import re
import shutil
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ARCHIVE = "https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123a/pandaset.zip"
PAPER_SEQUENCES = "001 011 016 028 053 063 084 106 123 158".split()
CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")


class HttpRangeFile(io.RawIOBase):
    """Seekable ZIP input with bounded caching and strict range validation."""

    block_size = 2 * 1024 * 1024

    def __init__(self, url, block_size=None):
        super().__init__()
        self.url = url
        if block_size is not None:
            self.block_size = block_size
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.blocks = OrderedDict()
        self.position = 0
        self.size = None
        self._request(0, 0)

    def _request(self, start, end):
        # Include the range in the cache key for intermediaries caching redirects.
        with self.session.get(
            self.url, params={"range": f"{start}-{end}"}, stream=True, timeout=(15, 90),
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        ) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match:
                raise OSError(f"HTTP Range unavailable ({response.status_code}); use the local ZIP fallback")
            first, last, size = map(int, match.groups())
            if (first, last) != (start, end) or self.size not in (None, size):
                raise OSError("Unexpected byte range or archive size")
            self.size = size
            data = response.content
            if len(data) != end - start + 1:
                raise OSError("Incomplete archive range; rerun this cell")
            return data

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise ValueError("Negative ZIP offset")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            block = self.position // self.block_size * self.block_size
            if block not in self.blocks:
                self.blocks[block] = self._request(block, min(self.size, block + self.block_size) - 1)
                if len(self.blocks) > 8:
                    self.blocks.popitem(last=False)
            self.blocks.move_to_end(block)
            next_position = min(end, block + len(self.blocks[block]))
            chunks.append(self.blocks[block][self.position - block:next_position - block])
            self.position = next_position
        return b"".join(chunks)

    def close(self):
        self.session.close()
        super().close()


def sequence_members(archive, sequence):
    if not re.fullmatch(r"\d{3}", sequence):
        raise ValueError(f"Invalid sequence ID: {sequence!r}")
    prefix = f"pandaset/{sequence}/"
    files = [info for info in archive.infolist() if info.filename.startswith(prefix) and not info.is_dir()]
    if not files:
        raise ValueError(f"Sequence {sequence} is absent from the archive")
    return sorted(files, key=lambda info: info.header_offset)


def sequence_ids(archive):
    return sorted({match[1] for info in archive.infolist()
                   if (match := re.fullmatch(r"pandaset/(\d{3})/annotations/cuboids/\d+\.pkl\.gz", info.filename))})


def read_cuboids(archive, sequence):
    prefix = f"pandaset/{sequence}/annotations/cuboids/"
    for info in sequence_members(archive, sequence):
        if info.filename.startswith(prefix) and info.filename.endswith(".pkl.gz"):
            yield int(PurePosixPath(info.filename).name.split(".")[0]), pd.read_pickle(
                io.BytesIO(archive.read(info)), compression="gzip"
            )


def shortlist(archive, sequences):
    """Inventory moving actors. Camera visibility and occlusion require image review."""
    records = []
    for sequence in sequences:
        for frame, boxes in read_cuboids(archive, sequence):
            boxes = boxes.loc[(~boxes["stationary"]) & (boxes["cuboids.sensor_id"] != 1)]
            missing = pd.Series("", index=boxes.index)
            walking = boxes["label"].isin(["Pedestrian", "Pedestrian with Object"]) & (
                boxes.get("attributes.pedestrian_behavior", missing) == "Walking"
            )
            cycling = (boxes["label"] == "Bicycle") & (boxes.get("attributes.rider_status", missing) == "With Rider")
            for row in boxes.loc[walking | cycling, ["uuid", "label", "camera_used"]].itertuples(index=False):
                records.append({"sequence": sequence, "frame": frame, "uuid": row.uuid,
                                "case": "cycling" if row.label == "Bicycle" else "walking",
                                "camera_hint": row.camera_used})
    columns = ["sequence", "case", "uuid", "frames", "camera_hints", "count"]
    if not records:
        return pd.DataFrame(columns=columns)
    rows = []
    for (sequence, case, uuid), group in pd.DataFrame(records).groupby(["sequence", "case", "uuid"]):
        frames = sorted(set(group.frame))
        rows.append([sequence, case, uuid, frames, sorted(set(group.camera_hint)), len(frames)])
    return pd.DataFrame(rows, columns=columns).sort_values(["count", "sequence", "uuid"], ascending=[False, True, True])


def sample_rgb(archive, sequence, camera, frame):
    from PIL import Image

    if camera not in CAMERAS or not 0 <= frame < 80:
        raise ValueError("Choose a PandaSet camera and a frame from 0 to 79")
    filename = f"pandaset/{sequence}/camera/{camera}_camera/{frame:02d}.jpg"
    with archive.open(filename) as source:
        image = Image.open(io.BytesIO(source.read())).convert("RGB")
    image.thumbnail((720, 480))
    return image


def pack_sequence(archive, sequence, target):
    """Create the same per-sequence ZIP from a remote or local source."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as subset:
            for info in sequence_members(archive, sequence):
                with archive.open(info) as source, subset.open(info.filename, "w") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def validate_sequence(root):
    import json
    import math

    root = Path(root)
    required = [root / "lidar" / name for name in ("poses.json", "timestamps.json")]
    for camera in CAMERAS:
        camera_root = root / "camera" / f"{camera}_camera"
        required.extend(camera_root / name for name in ("poses.json", "timestamps.json", "intrinsics.json"))
        required.extend(camera_root / f"{frame:02d}.jpg" for frame in range(80))
    for folder in ("lidar", "annotations/cuboids"):
        required.extend(root / folder / f"{frame:02d}.pkl.gz" for frame in range(80))
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: {len(missing)} missing files, including {missing[:8]}")
    folders = [root / "lidar", *(root / "camera" / f"{camera}_camera" for camera in CAMERAS)]
    for folder in folders:
        with (folder / "timestamps.json").open(encoding="utf-8") as source:
            times = json.load(source)
        with (folder / "poses.json").open(encoding="utf-8") as source:
            poses = json.load(source)
        if (len(times) != 80 or len(poses) != 80 or not all(map(math.isfinite, times))
                or any(a >= b for a, b in zip(times, times[1:]))):
            raise ValueError(f"Invalid timestamps or poses in {folder}")


def stage_sequence(drive_zip, data_root, sequence):
    """Copy to the runtime disk, safely extract one scene and validate layout."""
    data_root, drive_zip = Path(data_root), Path(drive_zip)
    target = data_root / sequence
    if target.exists():
        validate_sequence(target)
        return target
    if not drive_zip.is_file():
        raise FileNotFoundError(f"Upload pandaset_{sequence}.zip to {drive_zip.parent}")
    import tempfile

    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_root) as temporary:
        temporary = Path(temporary)
        local_zip = temporary / drive_zip.name
        shutil.copyfile(drive_zip, local_zip)
        with zipfile.ZipFile(local_zip) as archive:
            members = sequence_members(archive, sequence)
            for info in members:
                parts = PurePosixPath(info.filename).parts
                if ".." in parts or "\\" in info.filename or ":" in info.filename:
                    raise ValueError(f"Unsafe ZIP member: {info.filename}")
                archive.extract(info, temporary)
        source = temporary / "pandaset" / sequence
        validate_sequence(source)
        source.replace(target)
    return target
