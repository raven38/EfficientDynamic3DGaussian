exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/americano --port 6068 --ip 127.0.0.4 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/americano" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/cross-hands1 --port 6066 --ip 127.0.0.5 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/cross-hands1" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/espresso --port 6069 --ip 127.0.0.6 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/espresso" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/keyboard --port 6070 --ip 127.0.0.7 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/keyboard" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/split-cookie --port 6069 --ip 127.0.0.6 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/split-cookie" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/tamping --port 6070 --ip 127.0.0.7 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/tamping" &
wait

echo "Done"


exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $2 --model_path "$exp_name1/keyboard/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $2 --model_path "$exp_name1/espresso/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $2 --model_path "$exp_name1/cross-hands1/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $2 --model_path "$exp_name1/americano/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $2 --model_path "$exp_name1/split-cookie/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $2 --model_path "$exp_name1/tamping/"  &
wait
echo "Done"
exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/keyboard/"  
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/espresso/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/cross-hands1/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/americano/"   
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/split-cookie/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/tamping/"   
wait
echo "Done"
