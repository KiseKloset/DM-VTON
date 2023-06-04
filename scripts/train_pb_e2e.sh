python train_pb_e2e.py --project runs/train/PFAFN_align_mobile_clean-viton --name FS_PB_e2e_100 \
--device 2 --align_corners --batch_size 14 --workers 16 --lr 0.00005 \
--niter 50 --niter_decay 50 --save_period 10 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/Clean-VITON/VITON_traindata \
--pb_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_warp_100/weights/pb_warp_epoch_100.pt' \