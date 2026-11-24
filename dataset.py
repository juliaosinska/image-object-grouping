import csv
from PIL import Image
import numpy as np
import os
from mask_preparation import compute_mask
from objects_analysis import connected_component_labeling
from geometry import crop_object

# WŁASNE: load defined labels for textured vs smooth buttons
def load_labels(path):
    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels = {}
        for entry in reader:
            labels[(entry["file"], int(entry["row"]), int(entry["col"]))] = entry["category"]

    return labels

# WłASNE: formalize row splits for button images
def assign_grid_positions(label_image, n, row_gap=40):
    centroids = []
    
    for i in range(1, n + 1):
        y, x = np.nonzero(label_image == i)
        
        centroids.append((i, y.mean(), x.mean()))

    centroids = sorted(centroids, key=lambda o: o[1])

    rows = [[centroids[0]]]
    for previous, current in zip(centroids, centroids[1:]):
        if current[1] - previous[1] > row_gap:
            rows.append([])
        rows[-1].append(current)

    positions = {}
    for row_number, row in enumerate(rows, start=1):
        row = sorted(row, key=lambda o: o[2])
        for col_number, (i, cy, cx) in enumerate(row, start=1):
            positions[i] = (row_number, col_number)

    return positions

# WŁASNE: load dataset, extract objects
def load_dataset(folder, labels_path, mask_params, margin=5):
    labels = load_labels(labels_path)

    image_files = []
    objects = []

    for file_name in os.listdir(folder):
        if file_name.lower().endswith((".jpg", ".png")):
            image_files.append(file_name)
    
    image_files.sort()

    for file_name in image_files:
        image = Image.open(os.path.join(folder, file_name)).convert("RGB")
        image_array = np.array(image)

        mask, _ = compute_mask(image_array, **mask_params)

        label_image, n = connected_component_labeling(mask, 8)

        positions = assign_grid_positions(label_image, n)

        for i in range(1, n + 1):
            row, col = positions[i]
            category = labels.get((file_name, row, col))

            if category is None:
                print(f"Skipping object because of lack of the category defined: file '{file_name}', row '{row}', column '{col}'.")
                continue

            image_fragment, mask_fragment = crop_object(image_array, label_image, i, margin)
            
            objects.append({"id": len(objects) + 1, "file": file_name, "row": row, "col": col,
                "category": category, "image": image_fragment, "mask": mask_fragment})

    return objects