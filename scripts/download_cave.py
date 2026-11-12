"""Download and unpack the CAVE multispectral dataset into <data-root>/CAVE/.

Idempotent: skips the download if the zip is present and the unpack if the
32 scene folders already exist.
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    out = args.data_root / "CAVE"
    out.mkdir(parents=True, exist_ok=True)
    scenes = [d for d in out.iterdir() if d.is_dir() and d.name.endswith("_ms")]
    if len(scenes) == 32:
        print(f"CAVE already present: {len(scenes)} scenes in {out}")
        return

    zip_path = args.data_root / "complete_ms_data.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(zip_path)

    print(f"Unpacking {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    scenes = [d for d in out.iterdir() if d.is_dir() and d.name.endswith("_ms")]
    print(f"Done: {len(scenes)} scenes in {out}")


if __name__ == "__main__":
    main()
