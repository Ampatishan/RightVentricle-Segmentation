#!/bin/bash
export nnUNet_raw="/home/ampatish/scratch/RV/nnUNET_raw"
export nnUNet_preprocessed="/home/ampatish/scratch/RV/nnUNET_preprocessed"
export nnUNet_results="//home/ampatish/scratch/RV/nnUNEt_results"
export nnUNet_cropped_data="/home/ampatish/scratch/RV/nnUNET_cropped_data"  # optional, needed sometimes


DATASET_ID="001"  
CONFIG="3d_fullres"
PLANNER="nnUNetTrainer"  

# Launch training on different GPUs for folds 0 to 3
for FOLD in {0..3}; do
    CUDA_VISIBLE_DEVICES=$FOLD  nnUNet_n_proc_DA=0 nnUNetv2_train $DATASET_ID $CONFIG $FOLD &
done

wait
echo "All fold trainings started on separate GPUs."
