"""Run DeepSlice on one image; executed with the DeepSlice environment's Python.

    python deepslice_worker.py section.png result.json [--fast]
    python deepslice_worker.py --download

Must not import slicereg: the DeepSlice environment does not have it installed.
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KEYS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")


def main(argv):
    print("Loading DeepSlice model (the first run downloads the weights)...", flush=True)
    from DeepSlice import DSModel

    model = DSModel("mouse")
    if argv[:1] == ["--download"]:
        from DeepSlice.metadata import metadata_loader

        cfg = model.config["weight_file_paths"]["mouse"]["secondary"]
        metadata_loader.get_data_path(cfg, model.metadata_path)
        print("DeepSlice weights are ready.", flush=True)
        return
    image, out = argv[0], argv[1]
    ensemble = "--fast" not in argv
    print("Predicting atlas position...", flush=True)
    model.predict(image_list=[os.path.abspath(image)], ensemble=ensemble,
                  section_numbers=False)
    row = model.predictions.iloc[0]
    result = {k: float(row[k]) for k in KEYS}
    result.update(width=int(row["width"]), height=int(row["height"]))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
