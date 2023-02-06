#!/bin/bash

##You don't need to change next 4 exports
#Sets where to save hugging face caches
export HF_HOME=/project/relater/di/students/islpra4/cache/huggingface
export HF_DATASETS_CACHE="/project/relater/di/students/islpra4/cache/huggingface/datasets"
#Sets which environment to use
export PATH="/home/eugan/miniconda3/envs/hug/bin:$PATH"

export OMP_NUM_THREADS=4

#no. need to change unless you want to debug
pythonCMD="python -u -W ignore"


mkdir -p model

#CUDA_VISIBLE_DEVICES=1 This sets which GPU to use. Make sure the one you use is not used by others first
#in this example my code for this exercise is in the file trainer.py
CUDA_VISIBLE_DEVICES=0 $pythonCMD model.py 2>&1 | tee run-seq2seq.log
