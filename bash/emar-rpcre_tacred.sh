export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=$1 python3 main/emar.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name FewRel \
  --data_file data/data_with_marker.json \
  --relation_file data/id2rel.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name EMAR \
  --num_of_relation 80  \
  --cache_file data/fewrel_data.pt \
  --rel_per_task 8 \
  --aca 1 



export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=$1 python3 main/emar.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name FewRel \
  --data_file data/data_with_marker.json \
  --relation_file data/id2rel.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name EMAR \
  --num_of_relation 80  \
  --cache_file data/fewrel_data.pt \
  --rel_per_task 8 



export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=$1 python3 main/emar.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name TACRED \
  --data_file data/data_with_marker_tacred.json \
  --relation_file data/id2rel_tacred.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name EMAR \
  --num_of_relation 40  \
  --cache_file data/TACRED_data.pt \
  --rel_per_task 4 



export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=$1 python3 main/emar.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name TACRED \
  --data_file data/data_with_marker_tacred.json \
  --relation_file data/id2rel_tacred.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name EMAR \
  --num_of_relation 40  \
  --cache_file data/TACRED_data.pt \
  --rel_per_task 4 \
  --aca 1 



export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=$1 python3 main/rp_cre.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name TACRED \
  --data_file data/data_with_marker_tacred.json \
  --relation_file data/id2rel_tacred.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name RPCRE \
  --num_of_relation 40  \
  --cache_file data/TACRED_data.pt \
  --rel_per_task 4 \
  --aca 1 



export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=$1 python3 main/rp_cre.py \
  --memory_size 10 \
  --total_round 5 \
  --task_name TACRED \
  --data_file data/data_with_marker_tacred.json \
  --relation_file data/id2rel_tacred.json \
  --num_of_train 420 \
  --num_of_val 140 \
  --num_of_test 140 \
  --batch_size_per_step 32 \
  --exp_name RPCRE \
  --num_of_relation 40  \
  --cache_file data/TACRED_data.pt \
  --rel_per_task 4 
