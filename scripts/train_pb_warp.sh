python train_pb_warp.py --project runs/train/DM-VTON_demo --name Teacher_warp \
--device 0 --align_corners --batch_size 18 --workers 16 --lr 0.00005 \
--niter 50 --niter_decay 50 --save_period 20 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/VITON-Clean/VITON_traindata