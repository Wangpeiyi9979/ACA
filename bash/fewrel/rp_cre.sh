

export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=$1 python3 main/rp_cre.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name FewRel \
  --data_file data/data_with_marker.json \
  --relation_file data/id2rel.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name RPCRE \
  --num_of_relation 80  \
  --cache_file data/fewrel_data.pt \
  --rel_per_task 8
