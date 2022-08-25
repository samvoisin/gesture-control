# standard libraries
import os
import shutil
from pathlib import Path

data_dir = Path("./data/").resolve()
dataset_dir = data_dir / "senz3d_dataset"
raw_dataset_dir = dataset_dir / "acquisitions"

# set up new directory structure
restruct_dataset_dir = dataset_dir / "restructured"
os.mkdir(restruct_dataset_dir)
for i in range(1, 12):
    restruct_gest_path = restruct_dataset_dir / f"G{i}"
    os.mkdir(restruct_gest_path)

    for subj_dir in raw_dataset_dir.iterdir():
        raw_subj_gest_dir = subj_dir / f"G{i}"
        sg_files = raw_subj_gest_dir.glob("*color.png")
        for old_f in sg_files:
            new_f = restruct_gest_path / f"{subj_dir.stem.lower()}-{old_f.stem}.png"
            shutil.copyfile(old_f, new_f)
