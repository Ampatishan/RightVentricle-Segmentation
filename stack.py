import os
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

def contour_to_mask(contour, image_shape):
    """
    Converts contour points to a binary mask.

    Args:
        contour_x (list or array): x coordinates of contour.
        contour_y (list or array): y coordinates of contour.
        image_shape (tuple): (height, width) of output mask.

    Returns:
        np.ndarray: binary mask with 1 inside the contour and 0 outside.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)  # empty mask

    # Make an array of shape (N_points, 1, 2)
    contour = np.stack((contour[0], contour[1]), axis=-1).astype(np.int32)
    contour = contour.reshape((-1, 1, 2))

    # Fill the polygon (contour) on the mask
    cv2.fillPoly(mask, [contour], color=1)

    return mask


def stack_and_save_npy_to_nifti(root_dir, output_image_dir, output_label_dir):
    """
    Stack 2D images to generate 3D images

    Args : 
        root_dir(Path) : directory of the 2D images 
        output_image_dir(Path) : directory of the output 3D images
        output_label_dir(Path) : directory of the output 3D labelss

    Returns:
        None
    """

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Recursively walk through the root_dir
    for root, dirs, files in os.walk(root_dir):
        image_files = sorted([os.path.join(root, f) for f in files if f.endswith("_image.npy")])
        cont_files = sorted([os.path.join(root, f) for f in files if f.endswith("_cont.npy")])

        if not image_files or not cont_files:
            continue  # Skip if this folder doesn't have relevant data

        # Ensure matching file pairs (e.g., based on filename stem)
        image_dict = {os.path.basename(f).replace("_image.npy", ""): f for f in image_files}
        cont_dict = {os.path.basename(f).replace("_cont.npy", ""): f for f in cont_files}

        common_keys = sorted(set(image_dict.keys()) & set(cont_dict.keys()))
        if not common_keys:
            print(f"[Warning] No matching image-cont pairs found in: {root}")
            continue

        image_stack = []
        mask_stack = []
        for key in common_keys:
            image = np.load(image_dict[key])
            image_stack.append(image)
            contour = np.load(cont_dict[key])
            mask = contour_to_mask(contour,image.shape)
            mask_stack.append(mask)

        image_stack = np.stack(image_stack, axis=0)
        mask_stack = np.stack(mask_stack, axis=0)
        
        binary_mask = (mask_stack > 0).astype(np.uint8)  # Now only 0 and 1
        sample_id = os.path.basename(root)
        image_path = os.path.join(output_image_dir, f"{sample_id}_0000.nii.gz")
        label_path = os.path.join(output_label_dir, f"{sample_id}.nii.gz")

        nib.save(nib.Nifti1Image(image_stack, affine=np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(binary_mask, affine=np.eye(4)), label_path)

        print(f"[Saved] {image_path}, {label_path}")