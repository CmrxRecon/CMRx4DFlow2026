# FlowVN
#CUDA_VISIBLE_DEVICES=2 python main.py --mode test --training prosp --loss supervised --root_dir ./data/philips_aorta --input rawdata_full_R8.npy --network FlowVN --ckpt_path ./weights/aorta_R8_flowvn.ckpt --features_in 1 --D_size 15 --T_size -1 --num_act_weights 71 --features_out 8 --kernel_size 5 --num_stages 10

# FlowMRI_Net
#CUDA_VISIBLE_DEVICES=2 python main.py --mode test --training prosp --loss ssdu --root_dir ./data/philips_aorta --input rawdata_full_R8.npy --network FlowMRI_Net --ckpt_path ./weights/aorta_R8_flowmri_net.ckpt --features_in 4 --D_size 1 --T_size -1 --features_out 25 --num_stages 10

# CS-LLR
#CUDA_VISIBLE_DEVICES=2 python utils/CS_LLR_exec.py --lamb_tv 0 --lamb_llr 0.25 --input_dir ./data/siemens_brain/ --input rawdata_full.npy --output_dir ./results --vol vol010
