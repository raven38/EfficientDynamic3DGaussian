for d in `ls -d data/dynerf/*/`;
do
    for v in `ls $d*.mp4`;
    do
	f="${v##*/}"
	python3 generate_flow.py --dataset_path $d --input_dir ${f%.mp4} --model raft-sintel.pth 
    done
done	 
