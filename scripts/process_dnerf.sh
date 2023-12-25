exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/lego --port 6068 --ip 127.0.0.4 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/lego" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/bouncingballs --port 6066 --ip 127.0.0.5 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/bouncingballs" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/jumpingjacks --port 6069 --ip 127.0.0.6 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/jumpingjacks" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/trex --port 6070 --ip 127.0.0.7 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/trex" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/mutant --port 6068 --ip 127.0.0.8 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/mutant" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/standup --port 6066 --ip 127.0.0.9 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/standup" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dnerf/hook --port 6069 --ip 127.0.0.10 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/hook" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/hellwarrior --port 6070 --ip 127.0.0.11 --lambda_flow 0 --lambda_lasso 0 --model_path "$exp_name1/hellwarrior" &
wait
echo "Done"


exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "$exp_name1/standup/"  --skip_train &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "$exp_name1/jumpingjacks/"  --skip_train &
wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "$exp_name1/bouncingballs/"  --skip_train &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "$exp_name1/lego/"  --skip_train &
wait
export CUDA_VISIBLE_DEVICES=4&&python render.py --model_path "$exp_name1/hellwarrior/"  --skip_train &
export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path "$exp_name1/hook/"  --skip_train &
wait
export CUDA_VISIBLE_DEVICES=6&&python render.py --model_path "$exp_name1/trex/"  --skip_train &
export CUDA_VISIBLE_DEVICES=7&&python render.py --model_path "$exp_name1/mutant/"  --skip_train &
wait
echo "Done"
exp_name1=$1

export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/standup/"  
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/jumpingjacks/" 
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/bouncingballs/" 
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/lego/"   

export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/hellwarrior/"  
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/hook/" 
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/trex/" 
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "$exp_name1/mutant/"   
wait
echo "Done"
