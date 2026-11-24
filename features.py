import numpy as np
import pandas as pd
from geometry import object_orientation, rotate_object, shape_descriptors
from texture import texture_descriptors

# WŁASNE: feature table
def build_feature_table(objects):
    rows = []

    for o in objects:
        image = o["image"]
        mask = o["mask"]

        theta = object_orientation(mask)
        _, mask_rotated = rotate_object(image, mask, theta)

        geometric = shape_descriptors(mask, mask_rotated)

        texture = texture_descriptors(image, mask)

        rows.append({"id": o["id"], "file": o["file"], "category": o["category"], **geometric, **texture})

    return pd.DataFrame(rows)
