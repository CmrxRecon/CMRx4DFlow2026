# FlowVN
#CUDA_VISIBLE_DEVICES=2 python main.py --mode train --training prosp --loss supervised --root_dir ./data/philips_aorta --input rawdata_full_R8.npy --network FlowVN --features_in 1 --D_size 15 --T_size 15 --num_act_weights 71 --features_out 8 --kernel_size 5 --num_stages 10 --epoch 50 --lr 0.0001 --batch_size 1

# FlowMRI_Net
#CUDA_VISIBLE_DEVICES=2 python main.py --mode train --training prosp --loss ssdu --root_dir ./data/siemens_brain --input rawdata_full_R8.npy --network FlowMRI_Net --features_in 4 --D_size 1 --T_size -1 --features_out 25 --num_stages 10 --epoch 50 --lr 0.0005 --batch_size 1  
