"""Convert explicitly identified legacy RE10K manifests from mislabeled w2c to c2w.
RGB, depth and latent assets are unchanged. Never run on a generic c2w manifest.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-w2c", action="store_true", required=True,
                        help="Explicitly assert the source is the legacy DA3 w2c extraction")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Choose a new output path; input and existing outputs are preserved")
    tmp = output.with_name(output.name + ".tmp")
    count = 0
    with source.open() as fi, tmp.open("x") as fo:
        for line in fi:
            record = json.loads(line)
            if record.get("camera_convention") is not None or record.get("camera_conversion") is not None:
                raise ValueError("Source already has a camera convention marker; refusing double inversion")
            for view in record["views"]:
                matrix = np.asarray(view["c2w"], dtype=np.float64)
                if matrix.shape == (3, 4):
                    matrix = np.vstack([matrix, [0, 0, 0, 1]])
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError("Invalid camera matrix")
                camera_to_world = np.linalg.inv(matrix)
                if not np.allclose(matrix @ camera_to_world, np.eye(4), atol=1e-7):
                    raise ValueError("Invalid camera inverse")
                view["c2w"] = camera_to_world.tolist()
            record["camera_convention"] = "c2w"
            record["camera_conversion"] = "legacy_DA3_w2c_inverted_v1"
            fo.write(json.dumps(record) + "\n")
            count += 1
    os.replace(tmp, output)
    print(f"Converted {count} scenes -> {output}")


if __name__ == "__main__":
    main()
