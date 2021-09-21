#!/bin/bash

# Run with mpiexec.hydra -f $HOSTFILE -np 4 -ppn 1 ./run_multinode_trainer.sh > multi_4nodes.out 2> multi_4nodes.err
HOSTFILE=~/hostfolder/${SLURM_NODELIST}_hostfile
scontrol show hostnames $SLURM_NODELIST > $HOSTFILE
cat $HOSTFILE

NGPUS=4
NNODES=$(wc -l < $HOSTFILE)
MASTER_NODE=$(head -n 1 $HOSTFILE)
LOCAL_RANK=$PMI_RANK
echo $LOCAL_RANK
echo $NNODES
echo $MASTER_NODE

LANG=java   
data_prefix=/scratch1/08401/ywen/data/c2c_data
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=save/gptneo-125M_8e-5_4nodes_acc2_wd1e-4
PRETRAINDIR=EleutherAI/gpt-neo-125M
LOGFILE=${OUTPUTDIR}/trainer.log
GRAD_ACC=2
python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --nnodes=$NNODES \
    --node_rank=$LOCAL_RANK \
    --master_addr=$MASTER \
  run_gptneo.py \
    --train_filename ${data_prefix}/context_data.final,${data_prefix}/body_data.final \
    --dev_filename ${data_prefix}/context_data.final.100,${data_prefix}/body_data.final.100 \
    --lit_file=$LITFILE \
    --langs=$LANG \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --gradient_accumulation_steps=$GRAD_ACC \
    --model_type=gpt-neo \
    --block_size=1024 \
    --warmup_steps=1000 \
    --do_train \
    --do_eval \
    --learning_rate=8e-5 \
    --weight_decay=1e-4 \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --num_train_epochs=1 \
    --logging_steps=50 \
    --save_steps=2000 \
    --eval_steps=50 \
    --report_to=wandb \
    --seed=666
    #--deepspeed=deepspeed_config.json \
    #--fp16