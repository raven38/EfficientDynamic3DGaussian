exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/aleks-teapot --port 6068 --ip 127.0.0.4 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/aleks-teapot" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/chickchicken --port 6066 --ip 127.0.0.5 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/chickchicken" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/cut-lemon1 --port 6069 --ip 127.0.0.6 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/cut-lemon1" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/hand1-dense-v2 --port 6070 --ip 127.0.0.7 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/hand1-dense-v2" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/hypernerf/slice-banana --port 6069 --ip 127.0.0.6 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/slice-banana" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/hypernerf/torchocolate --port 6070 --ip 127.0.0.7 --lambda_lasso 0 --lambda_flow 0 --approx_l $2 --model_path "$exp_name1/torchocolate" &
wait

echo "Done"


exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $2 --model_path "$exp_name1/hand1-dense-v2/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $2 --model_path "$exp_name1/cut-lemon1/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $2 --model_path "$exp_name1/chickchicken/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $2 --model_path "$exp_name1/aleks-teapot/"  &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $2 --model_path "$exp_name1/slice-banana/"  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $2 --model_path "$exp_name1/torchocolate/"  &
wait
echo "Done"
exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/hand1-dense-v2/"  
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/cut-lemon1/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/chickchicken/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/aleks-teapot/"   
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/slice-banana/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/torchocolate/"   
wait
echo "Done"
