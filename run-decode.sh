export HF_HOME=/project/relater/di/students/islpra4/cache/huggingface
export HF_DATASETS_CACHE="/project/relater/di/students/islpra4/cache/huggingface/datasets"
export PATH="/home/eugan/miniconda3/envs/hug/bin:$PATH"

pythonCMD="python -u -W ignore"

stm=test.stm
CUDA_VISIBLE_DEVICES=5 $pythonCMD decode.py \
                        --model model/checkpoint-46000 \
                        --processor model/ \
			--input-name "input_features" \
                        --stm $stm \
                        --vocab vocab_dict.json
