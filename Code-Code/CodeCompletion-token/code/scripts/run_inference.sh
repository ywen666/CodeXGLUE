LANG=java                       # set python for py150
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=save/gptneo-125M_inference
PRETRAINDIR=EleutherAI/gpt-neo-125M
LOAD_PATH=save/gptneo-125M_8e-5_4nodes_acc2_wd1e-4/checkpoint-10000
data_prefix=/scratch1/08401/ywen/data/c2c_data
LOGFILE=${OUTPUTDIR}/c2c_inference.log
CUDA_VISIBLE_DEVICES=0 python run_gptneo.py \
    --dev_filename ${data_prefix}/context_data.final.100,${data_prefix}/body_data.final.100 \
    --lit_file=$LITFILE \
    --load_name=$LOAD_PATH \
    --langs=$LANG \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --model_type=gpt-neo \
    --block_size=1024 \
    --do_test \
    --per_gpu_eval_batch_size=1 \
    --beam_size 6 \
    --seed=666