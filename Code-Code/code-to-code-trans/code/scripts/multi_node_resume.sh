#!/bin/bash

export WANDB_PROJECT=gptneo
export UCX_TLS=knem,dc_x,rc
LOCAL_RANK=$PMI_RANK

NGPUS=1
NNODES=1
MASTER=""
PARTITION=0
MVAPICH=false

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        --help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  --help           Display this help message"
            echo "  --ngpus [count]  Number of GPUs per node (default: 1)"
            echo "  --nnodes [count] Number of nodes this script is launched on (default: 1)"
            echo "  --master [addr]  Address of master node (default: \"\")"
            echo "  --partition      Which data partition to use"
            exit 0
        ;;
        --ngpus)
            NGPUS=$VALUE
        ;;
        --nnodes)
            NNODES=$VALUE
        ;;
        --master)
            MASTER=$VALUE
        ;;      
        --partition)
            PARTITION=$VALUE
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done

export WANDB_PROJECT=gptneo
export UCX_TLS=knem,dc_x,rc
LOCAL_RANK=$PMI_RANK

source ~/.bashrc
conda activate dl 
which python

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, using_mvapich=$MVAPICH, partition=$PARTITION

data_prefix=/scratch1/08401/ywen/data/c2c_data
#output_dir=gptneo-1.3B_8e-5_8nodes
#output_dir=gptneo-1.3B_8e-5_4nodes
#pretrained_model=EleutherAI/gpt-neo-1.3B
pretrained_model=EleutherAI/gpt-neo-125M
#output_dir=gptneo-125M_8e-5_fp32
output_dir=gptneo-125M_8e-5_2node_acc4_p${PARTITION}
echo $output_dir

python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --nnodes=$NNODES \
    --node_rank=$LOCAL_RANK \
    --master_addr=$MASTER \
  run_trainer.py \
	--do_train \
	--model_type gpt-neo \
	--model_name_or_path $pretrained_model \
	--train_filename ${data_prefix}/context_data.final,${data_prefix}/body_data.final \
	--dev_filename ${data_prefix}/context_data.final.100,${data_prefix}/body_data.final.100 \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
    --warmup_steps 500 \
	--beam_size 5 \
	--train_batch_size 1 \
    --gradient_accumulation_steps 4 \
	--learning_rate 8e-5 \
	--train_steps 400000 \
    --deepspeed deepspeed_config.json \
    --report_to wandb \
    --partition $PARTITION \
    --fp16 \
    --load_model_path gptneo-125M_8e-5_multinode/checkpoint-9000/pytorch_model.bin
