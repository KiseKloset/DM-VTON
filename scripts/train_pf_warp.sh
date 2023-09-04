python train_pf_warp.py --project runs/train/DM-VTON_demo --name Student_warp \
--device 0 --align_corners --batch_size 8 --workers 16 --lr 0.00005 \
--niter 50 --niter_decay 50 --save_period 20 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/VITON-Clean/VITON_traindata \
--pb_warp_checkpoint runs/train/DM-VTON_demo/Teacher_e2e/weights/pb_warp_last.pt \
--pb_gen_checkpoint runs/train/DM-VTON_demo/Teacher_e2e/weights/pb_gen_last.pt