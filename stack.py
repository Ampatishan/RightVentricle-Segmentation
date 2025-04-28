import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

def stack_and_save_npy_to_nifti(root_dir, output_image_dir, output_label_dir):
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
        cont_stack = []
        for key in common_keys:
            image_stack.append(np.load(image_dict[key]))
            cont_stack.append(np.load(cont_dict[key]))

        image_stack = np.stack(image_stack, axis=0)
        cont_stack = np.stack(cont_stack, axis=0)
        
        binary_mask = (cont_stack > 0).astype(np.uint8)  # Now only 0 and 1
        sample_id = os.path.basename(root)
        image_path = os.path.join(output_image_dir, f"{sample_id}_0000.nii.gz")
        label_path = os.path.join(output_label_dir, f"{sample_id}.nii.gz")

        nib.save(nib.Nifti1Image(image_stack, affine=np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(binary_mask, affine=np.eye(4)), label_path)

        print(f"[Saved] {image_path}, {label_path}")

# Example usage
stack_and_save_npy_to_nifti(
    root_dir="/home/ampatish/scratch/RV/2025a", 
    output_image_dir="/home/ampatish/scratch/RV/nnUNET_raw/Dataset001_RV/imagesTr", 
    output_label_dir="/home/ampatish/scratch/RV/nnUNET_raw/Dataset001_RV/labelsTr"
)
