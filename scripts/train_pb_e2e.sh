python train_pb_e2e.py --project runs/SRMGN_align_merge-viton-v1 --name PB_e2e_100 \
--device 0 --align_corners --batch_size 12 --workers 16 --lr 0.00004 \
--niter 10 --niter_decay 30 --save_period 5 \
--print_step 100 --sample_step 100 \
--resume \
--dataroot ../dataset/Merge-VITON-V1/VITON_traindata \
--pb_warp_checkpoint 'checkpoints/SRMGN_PB_e2e_align_merge-viton-v1_100/PBAFN_warp_epoch_061.pth' \
--pb_gen_checkpoint 'checkpoints/SRMGN_PB_e2e_align_merge-viton-v1_100/PBAFN_gen_epoch_061.pth'