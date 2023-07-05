python train_pf_warp.py --project  runs/train/PFAFN_align_mobile_warpKDloss_clean-viton --name PFAFN_PF_warp_align_mobile_warpKDloss_100 \
--device 1 --align_corners --batch_size 8 --workers 16 --lr 0.00005 \
--niter 50 --niter_decay 50 --save_period 20 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/Clean-VITON/VITON_traindata \
--pb_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_e2e_100/weights/pb_warp_last.pt' \
--pb_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_e2e_100/weights/pb_gen_last.pt'