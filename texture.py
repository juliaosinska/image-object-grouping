from PIL import Image
import numpy as np

# WŁASNE: gray-level co-occurrence matrix inside the object mask (distance 1, direction: right)
# GENAI: Claude – explaining the GLCM idea and the step-by-step plan; the code was written by me
def glcm(gray, mask, levels=8):
    # 1. quantization
    q = gray.astype(int) * levels // 256

    # 2. pairs (pixel, right neighbour)
    # all pixels except the last column
    left = q[:, :-1]
    # its right side neighbour
    right = q[:, 1:]

    valid = mask[:, :-1] & mask[:, 1:]
    i = left[valid]
    j = right[valid]

    # 3. counting the pairs -> levels x levels matrix
    pair_index = i * levels + j 

    counts = np.bincount(pair_index, minlength=levels * levels).reshape(levels, levels)
    
    # 4. normalization -> P(i, j)
    probability = counts / counts.sum()

    return probability

# WŁASNE: texture descriptors from the GLCM
def texture_descriptors(image_rgb, mask, levels=8):
    # 5. gray image of the crop
    gray = np.array(Image.fromarray(image_rgb).convert("L"))
    
    # 6. glcm
    P = glcm(gray, mask, levels)
    
    # 7. contrast and homogeneity
    I, J = np.indices((levels, levels))

    contrast = np.sum(P * (I - J) ** 2)
    homogeneity = np.sum(P / (1 + (I - J) ** 2))

    return {"contrast": float(contrast), "homogeneity": float(homogeneity)}

    