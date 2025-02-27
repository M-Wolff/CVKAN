"""
File: json_editor.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Manipulate the results.json file
"""
import datetime
import json
from pathlib import Path
import shutil


if __name__ == "__main__":
    # input results.json
    p_in = Path("/src/experiments/results.json")
    # create backup in results_backup folder
    p_backup = Path(
        f"/src/experiments/results_backups/results_backup_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    p_backup.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(p_in, p_backup)
    # read original json
    with open(p_in, "r", encoding="utf-8") as f:
        results = json.load(f)

    # modify results
    results_modified = []
    # iterate through all results (all experiments as list entries in results.json file)
    for result in results:
        # filter as desired. Everything not appended to results_modified is deleted
        #if not result["dataset_name"].startswith("ff_"):
        #if not (result["model_name"] == "CVKANWrapper" and result["dataset_name"].startswith("ff_")):
        #if not (result["model_name"] == "CVKANWrapper" and result["dataset_name"] == "knot_complex" and result["use_norm"][0] in ["batchnorm", "batchnormvar"]):
        if not (result["model_name"]=="CVKANWrapper" and result["dataset_name"]=="ph_holo_c_100k"):
            results_modified.append(result)
        else:
            print("deleting", result)
    # overwrite modified results.json
    with open(p_in, "w", encoding="utf-8") as f:
        json.dump(results_modified, f)
