export HF_HOME=/project/relater/di/students/islpra4/cache/huggingface
export HF_DATASETS_CACHE="/project/relater/di/students/islpra4/cache/huggingface/datasets"
export PATH="/home/eugan/miniconda3/envs/hug/bin:$PATH"

pythonCMD="python -u -W ignore"

stm=train.stm
CUDA_VISIBLE_DEVICES=0 $pythonCMD lm.py \
                        --stm $stm
