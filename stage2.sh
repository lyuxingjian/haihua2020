python stage1.py --experiment_name final_se50_0_stage1 --model_name se_resnext50_32x4d --lr 1e-1 --total_epochs 5 --batch_size 64 \
    --pretrained checkpoints/se50_0.pt --gridmask_ratio .6 --fold 0 --gpu_no 0
python stage1.py --experiment_name final_se50_1_stage1 --model_name se_resnext50_32x4d --lr 1e-1 --total_epochs 5 --batch_size 64 \
    --pretrained checkpoints/se50_1.pt --gridmask_ratio .6 --fold 2 --gpu_no 0

python stage1.py --experiment_name final_b4_0_stage1 --model_name efficientnet-b4 --lr 1e-1 --total_epochs 8 --batch_size 64 \
    --pretrained checkpoints/b4_0.pt --gridmask_ratio .6 --fold 1 --gpu_no 1
python stage1.py --experiment_name final_b4_1_stage1 --model_name efficientnet-b4 --lr 1e-1 --total_epochs 8 --batch_size 64 \
    --pretrained checkpoints/b4_1.pt --gridmask_ratio .6 --fold 3 --gpu_no 1

python stage1.py --experiment_name final_iv4_0_stage1 --model_name inceptionv4 --lr 5e-2 --total_epochs 10 --batch_size 64 \
    --pretrained checkpoints/iv4_0.pt --gridmask_ratio .6 --fold 3 --gpu_no 0
python stage1.py --experiment_name final_iv4_1_stage1 --model_name inceptionv4 --lr 5e-2 --total_epochs 10 --batch_size 64 \
    --pretrained checkpoints/iv4_1.pt --gridmask_ratio .6 --fold 2 --gpu_no 0
python stage1.py --experiment_name final_iv4_2_stage1 --model_name inceptionv4 --lr 5e-2 --total_epochs 10 --batch_size 64 \
    --pretrained checkpoints/iv4_2.pt --gridmask_ratio .6 --fold 0 --gpu_no 0

python stage1.py --experiment_name final_mixxl_0_stage1 --model_name mixnet_xl --lr 1e-1 --total_epochs 8 --batch_size 64 \
    --pretrained checkpoints/mixxl_0.pt --gridmask_ratio .6 --fold 4 --gpu_no 1
python stage1.py --experiment_name final_mixxl_1_stage1 --model_name mixnet_xl --lr 1e-1 --total_epochs 8 --batch_size 64 \
    --pretrained checkpoints/mixxl_1.pt --gridmask_ratio .6 --fold 1 --gpu_no 1