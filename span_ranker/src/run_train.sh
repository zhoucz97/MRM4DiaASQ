
data_dir="../../data"
task="qiaasq"
lan="zh"
plm_model="roberta-large"
plm_name="roberta_large"

if [ $lan != 'en' ]
then
plm_model="hfl/chinese-roberta-wwm-ext"
plm_name="chinese_roberta_wwm_ext"
fi


python train.py \
  --mode train \
  --bert_model $plm_model \
  --data_dir $data_dir \
  --lan $lan \
  --output_dir $data_dir/saves/ranker \
  --train_path $task/jsons_${lan}/train.json \
  --valid_path $task/jsons_${lan}/valid.json \
  --test_path $task/jsons_${lan}/test.json \
  --top_k 128 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --learning_rate 5e-5 \
  --plm_lr 1e-5 \
  --weight_decay 1e-2 \
  --warmup_proportion 0.1 \
  --max_seq_len 512 \
  --span_max_len 10 \
  --num_epochs 15 \
  --seed 42 \

  
