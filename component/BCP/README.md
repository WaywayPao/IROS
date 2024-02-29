## **Environment set up**
	pip install -r requirements.txt

## **Generate tracklet**
	python utils/gen_tracking.py	

## **Training**
	python train.py --data_type all --gpu DEVICE_ID
	
## **Testing**
	python test.py --data_type {DATA_TYPE} --ckpt_path {PATH}

## **See traing log**
	tensorboard dev upload --logdir logs/{LOG_NAME} --name {LOG_NAME}

## TBD
	python train.py --method vision --batch_size 8 --lr 0.0001 --use_target_point --gpu 0 --verbose
	python train.py --method bev_seg --batch_size 8 --lr 0.00001 --use_target_point --gpu 0 --verbose
	python train.py --method pf --batch_size 8 --lr 0.00001 --use_target_point --gpu 0 --verbose
	
	python test_roi.py --method pf --batch_size 32 --use_target_point --gpu 0 --verbose --ckpt_path ./ckpts/2024-1-25_161427/epoch-10.pth --save_roi

	python test.py --method vision --use_target_point --gpu 0 --verbose --ckpt_path ./ckpts/2024-1-23_190536/epoch-7.pth --save_roi

	# 2/27
	python train.py --method pf --batch_size 32 --num_worker 8 --lr 0.0000005 --gpu 0 --verbose --ckpt_path "./checkpoints/2024-2-25_002941/epoch-40.pth" --epochs 40

	# gt pf
	2024-2-28_210146 epoch-5
	
	# not-gt + "./checkpoints/2024-2-28_224123/epoch-5.pth"
	2024-2-29_001439 epoch-39
	