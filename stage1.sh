python stage0.py --experiment_name final_se50_0 --model_name se_resnext50_32x4d --lr 2e-1 --total_epochs 20 --batch_size 64 --gridmask_ratio .6 --fold 0 --gpu_no 0
python stage0.py --experiment_name final_se50_1 --model_name se_resnext50_32x4d --lr 2e-1 --total_epochs 20 --batch_size 64 --gridmask_ratio .6 --fold 2 --gpu_no 1

python stage0.py --experiment_name final_b4_0 --model_name efficientnet-b4 --lr 2e-1 --total_epochs 30 --batch_size 64 --gridmask_ratio .6 --fold 1 --gpu_no 0
python stage0.py --experiment_name final_b4_1 --model_name efficientnet-b4 --lr 2e-1 --total_epochs 30 --batch_size 64 --gridmask_ratio .6 --fold 3 --gpu_no 1

python stage0.py --experiment_name final_iv4_0 --model_name inceptionv4 --lr 1e-1 --total_epochs 40 --batch_size 64 --gridmask_ratio .6 --fold 3 --gpu_no 0
python stage0.py --experiment_name final_iv4_1 --model_name inceptionv4 --lr 1e-1 --total_epochs 40 --batch_size 64 --gridmask_ratio .6 --fold 2 --gpu_no 1
python stage0.py --experiment_name final_iv4_2 --model_name inceptionv4 --lr 1e-1 --total_epochs 40 --batch_size 64 --gridmask_ratio .6 --fold 0 --gpu_no 1