export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/bouncingballs --port 6066 --ip 127.0.0.5 --lambda_flow 100 --model_path "search_flow_lambda/bouncingballs_fl100"
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/bouncingballs --port 6066 --ip 127.0.0.5 --lambda_flow 10 --model_path "search_flow_lambda/bouncingballs_fl10"
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/bouncingballs --port 6066 --ip 127.0.0.5 --lambda_flow 50 --model_path "search_flow_lambda/bouncingballs_fl50"
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/bouncingballs --port 6066 --ip 127.0.0.5 --lambda_flow 200 --model_path "search_flow_lambda/bouncingballs_fl200"
export CUDA_VISIBLE_DEVICES=1&&python train.py -s data/dnerf/bouncingballs --port 6066 --ip 127.0.0.5 --lambda_flow 400 --model_path "search_flow_lambda/bouncingballs_fl400"
echo "Done"


export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "search_flow_lambda/bouncingballs_fl100" --skip_train
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "search_flow_lambda/bouncingballs_fl10" --skip_train
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "search_flow_lambda/bouncingballs_fl50" --skip_train
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "search_flow_lambda/bouncingballs_fl200" --skip_train
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "search_flow_lambda/bouncingballs_fl400" --skip_train
echo "Done"

export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "search_flow_lambda/bouncingballs_fl100"
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "search_flow_lambda/bouncingballs_fl10"
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "search_flow_lambda/bouncingballs_fl50"
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "search_flow_lambda/bouncingballs_fl200"
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "search_flow_lambda/bouncingballs_fl400"

echo "Done"
