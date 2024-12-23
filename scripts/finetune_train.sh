#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption
# 2nd arg: path to target images
# 3rd arg: path where retrieved images are saved
# 4th arg: name of the experiment
# 5th arg: config name
# 6th arg: pretrained model path

ARRAY=()

for i in "$@"
do
    echo $i
    ARRAY+=("${i}")
done

if [ "${ARRAY[4]}" == "finetune.yaml" ]; then
    python -u train.py \
            --base configs/custom-diffusion/${ARRAY[4]}  \
            -t --gpus 0,1,2,3 \
            --resume-from-checkpoint-custom ${ARRAY[5]} \
            --caption "${ARRAY[0]}" \
            --datapath ${ARRAY[1]} \
            --reg_datapath "${ARRAY[2]}/images.txt" \
            --reg_caption "${ARRAY[2]}/caption.txt" \
            --name "${ARRAY[3]}-sdv4"

elif [ "${ARRAY[4]}" == "finetune_addtoken.yaml" ]; then
    python -u  train.py \
        --base configs/custom-diffusion/${ARRAY[4]}  \
        -t --gpus 0,1,2,3 \
        --resume-from-checkpoint-custom ${ARRAY[5]} \
        --caption "<new1> ${ARRAY[0]}" \
        --datapath ${ARRAY[1]} \
        --reg_datapath "${ARRAY[2]}/samples" \
        --reg_caption "${ARRAY[0]}" \
        --modifier_token "<new1>" \
        --name "${ARRAY[3]}-sdv4"
else
    python -u train.py \
            --base configs/custom-diffusion/${ARRAY[4]}  \
            -t --gpus 0,1,2,3 \
            --resume-from-checkpoint-custom ${ARRAY[5]} \
            --caption "<new1> ${ARRAY[0]}" \
            --datapath ${ARRAY[1]} \
            --reg_datapath "${ARRAY[2]}/images.txt" \
            --reg_caption "${ARRAY[2]}/caption.txt" \
            --modifier_token "<new1>" \
            --name "${ARRAY[3]}-sdv4"
fi
