exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/vrig-3dprinter --port 6068 --ip 127.0.0.4 --lambda_lasso 0 --lambda_flow 0 --model_path "$exp_name1/vrig-3dprinter" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/vrig-chicken --port 6066 --ip 127.0.0.5 --lambda_lasso 0 --lambda_flow 0 --model_path "$exp_name1/vrig-chicken" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/vrig-broom --port 6069 --ip 127.0.0.6 --lambda_lasso 0 --lambda_flow 0 --model_path "$exp_name1/vrig-broom" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/vrig-peel-banana --port 6070 --ip 127.0.0.7 --lambda_lasso 0 --lambda_flow 0 --model_path "$exp_name1/vrig-peel-banana" &
wait
echo "Done"


exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "$exp_name1/vrig-peel-banana/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "$exp_name1/vrig-broom/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "$exp_name1/vrig-chicken/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "$exp_name1/vrig-3dprinter/"  &
wait
echo "Done"
exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-peel-banana/"  
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-broom/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-chicken/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-3dprinter/"   
wait
echo "Done"
