exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python train.py -s data/dynerf/coffee_martini --port 6068 --ip 127.0.0.4 --lambda_flow $2 --lambda_lasso $3 --approx_l $5 --model_path "$exp_name1/coffee_martini" &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dynerf/cook_spinach --port 6066 --ip 127.0.0.5 --lambda_flow $2 --lambda_lasso $3 --approx_l $5 --model_path "$exp_name1/cook_spinach" &
export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dynerf/cut_roasted_beef --port 6069 --ip 127.0.0.6 --lambda_flow $2 --lambda_lasso $3 --approx_l $5 --model_path "$exp_name1/cut_roasted_beef" &
export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dynerf/flame_salmon_1 --port 6070 --ip 127.0.0.7 --lambda_flow $2 --lambda_lasso $3 --approx_l $5 --model_path "$exp_name1/flame_salmon_1" &
export CUDA_VISIBLE_DEVICES=4&&python train.py -s data/dynerf/flame_steak --port 6068 --ip 127.0.0.4 --lambda_flow $2 --lambda_lasso $3 --approx_l $5 --model_path "$exp_name1/flame_steak" &
export CUDA_VISIBLE_DEVICES=5&&python train.py -s data/dynerf/sear_steak --port 6066 --ip 127.0.0.5 --lambda_flow $2 --lambda_lasso $3 --approx_l $5 --model_path "$exp_name1/sear_steak" &
wait 
echo "Done"


exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "$exp_name1/sear_steak/"  --skip_train --approx_l $5 &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "$exp_name1/cut_roasted_beef/"  --skip_train --approx_l $5 &
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "$exp_name1/cook_spinach/"  --skip_train --approx_l $5 &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "$exp_name1/coffee_martini/"  --skip_train --approx_l $5 &
export CUDA_VISIBLE_DEVICES=4&&python render.py --model_path "$exp_name1/flame_salmon_1/"  --skip_train --approx_l $5 &
export CUDA_VISIBLE_DEVICES=5&&python render.py --model_path "$exp_name1/flame_steak/"  --skip_train --approx_l $5 &
wait
echo "Done"
exp_name1=$1

export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/sear_steak/"  
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/cut_roasted_beef/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/cook_spinach/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/coffee_martini/"   
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/flame_salmon_1/" 
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "$exp_name1/flame_steak/"   
wait
echo "Done"
