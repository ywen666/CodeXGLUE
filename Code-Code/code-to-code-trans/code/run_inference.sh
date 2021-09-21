output_dir=tacc_inference/gpt125M #rohan-context-model-1 #the place where you want to save the fine-tuned models and predictions
output_dir=tacc_inference/gpt1.3b #rohan-context-model-1 #the place where you want to save the fine-tuned models and predictions
pretrained_model=EleutherAI/gpt-neo-125M
pretrained_model=EleutherAI/gpt-neo-1.3B
LOAD_PATH=gptneo-125M_8e-5_4nodes_acc2_wd001/checkpoint-203000/pytorch_model.bin
LOAD_PATH=gptneo-1.3B_8e-5_16nodes_acc1_wd1e-4/checkpoint-23000/pytorch_model.bin
LOAD_PATH=gptneo-1.3B_8e-5_24nodes_acc1_wd1e-4/pytorch_model.bin
data_prefix=/scratch1/08401/ywen/data/c2c_data
##CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 run.py \
#CUDA_VISIBLE_DEVICES=0 python run.py \
#CUDA_VISIBLE_DEVICES=0 python run.py \
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --do_test \
    --model_type gpt-neo \
    --model_name_or_path $pretrained_model \
    --load_model_path $LOAD_PATH \
	--test_filename ${data_prefix}/context_data.final.100,${data_prefix}/body_data.final.100 \
    --output_dir $output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --beam_size 5 \
    --eval_batch_size 1 \
    --fp16
