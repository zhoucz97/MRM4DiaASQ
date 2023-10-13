

data_dir="../../data"
output_dir="../outputs/$cur_time/"
plm_dir=""
plm_name="roberta-large"
task="qiaasq"
lan="en"


if [ $lan != 'en' ]
then
plm_name="hfl/chinese-roberta-wwm-ext"
fi


python train.py \
  --mode train \
  --top_k 128 \
  --plm_name $plm_name \
  --output_dir $output_dir \
  --cache_path $data_dir/cache/$lan \
  --train_path $data_dir/$task/jsons_${lan}/train.json \
  --valid_path $data_dir/$task/jsons_${lan}/valid.json \
  --test_path $data_dir/$task/jsons_${lan}/test.json \
  --ranker_path "../../data/rankers/$lan" \
  --lan $lan \
  --batch_size 1 \
  --eval_batch_size 1 \
  --plm_lr 1e-5 \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --warmup_proportion 0.1 \
  --max_seq_len 512 \
  --num_epochs 35 \
  --seed_list 0 1 2 3 4 \
  --logging_steps 20 \
  --init_time $cur_time \
  --ckpt_path $ckpt_path