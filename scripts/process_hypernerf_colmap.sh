exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/colmap_hypernerf/vrig-3dprinter --port 6068 --ip 127.0.0.4 --lambda_lasso $2 --lambda_flow $3 --approx_l $4 --model_path "$exp_name1/vrig-3dprinter" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/colmap_hypernerf/vrig-chicken --port 6066 --ip 127.0.0.5 --lambda_lasso $2 --lambda_flow $3 --approx_l $4 --model_path "$exp_name1/vrig-chicken" &
wait
export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/colmap_hypernerf/broom2 --port 6069 --ip 127.0.0.6 --lambda_lasso $2 --lambda_flow $3 --approx_l $4 --model_path "$exp_name1/vrig-broom" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/colmap_hypernerf/vrig-peel-banana --port 6070 --ip 127.0.0.7 --lambda_lasso $2 --lambda_flow $3 --approx_l $4 --model_path "$exp_name1/vrig-peel-banana" &
wait
echo "Done"


exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $4 --model_path "$exp_name1/vrig-peel-banana/"  --skip_train &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $4 --model_path "$exp_name1/vrig-broom/"  --skip_train &
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --approx_l $4 --model_path "$exp_name1/vrig-chicken/"  --skip_train &
export CUDA_VISIBLE_DEVICES=1&&python render.py --approx_l $4 --model_path "$exp_name1/vrig-3dprinter/"  --skip_train &
wait
echo "Done"
exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-peel-banana/"
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-broom/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-chicken/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/vrig-3dprinter/"   
wait
echo "Done"
