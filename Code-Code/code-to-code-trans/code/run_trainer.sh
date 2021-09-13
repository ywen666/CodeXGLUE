pretrained_model=EleutherAI/gpt-neo-1.3B
output_dir=gptneo-1.3B_8e-5 #the place where you want to save the fine-tuned models and predictions
#output_dir=gptneo-1.3B_5e-4 #the place where you want to save the fine-tuned models and predictions
#output_dir=gptneo-1.3B_5e-4 #the place where you want to save the fine-tuned models and predictions
#CUDA_LAUNCH_BLOCKING=1 python run.py \
#CUDA_VISIBLE_DEVICES=1,2,3 python run.py \
#CUDA_VISIBLE_DEVICES=1 python run.py \
#CUDA_VISIBLE_DEVICES=0 python run.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py \
#CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_trainer.py \
#CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
#CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
#deepspeed --include localhost:0,1,2,3 run_trainer.py \
#deepspeed --include localhost:1,2,3 run_trainer.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_trainer.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_trainer.py \
	--do_train \
	--model_type gpt-neo \
	--model_name_or_path $pretrained_model \
	--train_filename /home/ewen/data/c2c_data/context_data.final,/home/ewen/data/c2c_data/body_data.final \
	--dev_filename /home/ewen/data/c2c_data/context_data.final.100,/home/ewen/data/c2c_data/body_data.final.100 \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
    --warmup_steps 2000 \
	--beam_size 5 \
	--train_batch_size 1 \
    --gradient_accumulation_steps 8 \
	--eval_batch_size 1 \
	--learning_rate 8e-5 \
	--train_steps 400000 \
	--eval_steps 1000
    #--deepspeed deepspeed_config.json \
    #--fp16 

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