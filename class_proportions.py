from glob import glob
import os

import laspy
import numpy as np

def class_proportions(
        input_dir: str, ignored_classes: list[int]) -> dict[int, float]:
    counts = {}
    total = 0
    for fp in glob(os.path.join(input_dir, "*.la[sz]")):
        print(f"Processing {os.path.basename(fp)}")
        las = laspy.read(fp)
        for label in np.unique(las.classification):
            if label in ignored_classes:
                continue
            count = np.sum(las.classification == label)
            if label in counts.keys():
                counts[label] += count
            else:
                counts[label] = count
            total += count

    return {c: counts[c] / total for c in counts.keys()}

if __name__ == "__main__":
    input_dir = r"C:\Users\BEBLADES\data\dales\train"
    ignored_classes = []

    print(class_proportions(input_dir, ignored_classes))