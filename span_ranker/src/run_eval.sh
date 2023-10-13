
# 启用第i块GPU
export CUDA_VISIBLE_DEVICES=6


data_dir="../../data"
task="qiaasq"
lan="zh"
plm_name="roberta-large"
ckpt_path=$data_dir/saves/ranker/model_best.$lan.pt

if [ $lan != 'en' ]
then
plm_name="hfl/chinese-roberta-wwm-ext"
fi


python train.py \
  --mode test \
  --mode_type_list test valid train \
  --save_rank True \
  --top_k 64 \
  --lan $lan \
  --mode_save_path "$data_dir/saves/ranker" \
  --bert_model $plm_name \
  --data_dir $data_dir \
  --train_path $task/jsons_${lan}/train.json \
  --valid_path $task/jsons_${lan}/valid.json \
  --test_path $task/jsons_${lan}/test.json \
  --batch_size 1 \
  --eval_batch_size 1 \
  --learning_rate 5e-5 \
  --weight_decay 1e-2 \
  --warmup_proportion 0.1 \
  --max_seq_len 512 \
  --num_epochs 25 \
  --seed 42 \
  --logging_steps 200 \
  --ckpt_path $ckpt_path
