python train_pf_warp.py --project runs/SRMGN_align_merge-viton-v1 --name PF_warp_100 \
--device 0 --align_corners --batch_size 10 --workers 16 \
--niter 50 --niter_decay 50 --save_period 10 \
--print_step 100 --sample_step 100 \
--dataroot ../dataset/Merge-VITON-V1/VITON_traindata \
--pb_warp_checkpoint 'runs/SRMGN_align_merge-viton-v1/PB_e2e_100/weights/pb_warp_last.pt' \
--pb_gen_checkpoint 'runs/SRMGN_align_merge-viton-v1/PB_e2e_100/weights/pb_gen_last.pt'