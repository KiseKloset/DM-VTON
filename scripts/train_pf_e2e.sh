python train_pf_e2e.py --project runs/SRMGN_align_merge-viton-v1 --name PF_e2e_100 \
--device 1 --align_corners --batch_size 3 --workers 16 \
--niter 50 --niter_decay 50 --save_period 1 \
--print_step 100 --sample_step 100 \
--dataroot ../dataset/Merge-VITON-V1/VITON_traindata \
--pb_warp_checkpoint 'runs/SRMGN_align_merge-viton-v1/PB_e2e_100/weights/pb_warp_last.pt' \
--pb_gen_checkpoint 'runs/SRMGN_align_merge-viton-v1/PB_e2e_100/weights/pb_gen_last.pt' \
--pf_warp_checkpoint 'runs/SRMGN_align_merge-viton-v1/PF_warp_100/weights/pf_warp_last.pt' 