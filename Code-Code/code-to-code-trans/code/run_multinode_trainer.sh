#!/bin/bash

# Run with mpiexec.hydra -f hostfile -np 4 -ppn 1 ./run_multinode_trainer.sh > multi_4nodes.out 2> multi_4nodes.err
#HOSTFILE=/tmp/hostfile
#scontrol show hostnames $SLURM_NODELIST > $HOSTFILE
#cat $HOSTFILE

GPU_PER_NODE=4
NODES=2
MASTER_NODE=c199-071

#LOCAL_RANK=$MV2_COMM_WORLD_RANK
#export MV2_USE_CUDA=1
#export MV2_ENABLE_AFFINITY=1
#export MV2_THREADS_PER_PROCESS=2
#export MV2_SHOW_CPU_BINDING=1
#export MV2_CPU_BINDING_POLICY=hybrid
#export MV2_HYBRID_BINDING_POLICY=spread
#export MV2_USE_RDMA_CM=0
#export MV2_SUPPORT_DL=1
#
#export OMP_NUM_THREADS=4
#export NCCL_IB_DISABLE=1
#export UCX_TLS=knem,dc_x,rc
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO

#export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export WANDB_PROJECT=gptneo
export UCX_TLS=knem,dc_x,rc
LOCAL_RANK=$PMI_RANK

pretrained_model=EleutherAI/gpt-neo-1.3B
data_prefix=/scratch1/08401/ywen/data/c2c_data
#output_dir=gptneo-1.3B_8e-5_multinode #the place where you want to save the fine-tuned models and predictions
output_dir=gptneo-1.3B_8e-5_4node #the place where you want to save the fine-tuned models and predictions
python -m torch.distributed.launch \
    --nproc_per_node=$GPU_PER_NODE \
    --nnodes=$NODES \
    --node_rank=$LOCAL_RANK \
    --master_addr=$MASTER_NODE \
  run_trainer.py \
	--do_train \
	--model_type gpt-neo \
	--model_name_or_path $pretrained_model \
	--train_filename ${data_prefix}/context_data.final,${data_prefix}/body_data.final \
	--dev_filename ${data_prefix}/context_data.final.100,${data_prefix}/body_data.final.100 \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
    --warmup_steps 2000 \
	--beam_size 5 \
	--train_batch_size 1 \
    --gradient_accumulation_steps 2 \
	--eval_batch_size 1 \
	--learning_rate 8e-5 \
	--train_steps 400000 \
    --deepspeed deepspeed_config.json \
    --report_to wandb \
    --fp16 

#output_dir=rohan-context-model-final-dropout #rohan-context-model-1 #the place where you want to save the fine-tuned models and predictions
#CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 run.py \
#    --do_test \
#    --model_type gptneo \
#    --model_name_or_path $pretrained_model \
#    --load_model_path $output_dir/checkpoint-best-ppl/pytorch_model.bin \
#	--test_filename /home/ewen/data/c2c_data/context_data.final.100,/home/ewen/data/c2c_data/body_data.final.100 \
#    --output_dir $output_dir \
#    --max_source_length 512 \
#    --max_target_length 512 \
#    --beam_size 1 \
#    --eval_batch_size  8 
