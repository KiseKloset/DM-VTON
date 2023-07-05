python train_pf_all.py --project  runs/train/PFAFN_align_mobile_all_clean-viton --name PFAFN_PF_align_mobile_all_noedge \
--device 1 --align_corners --batch_size 6 --workers 16 --lr 0.00005 \
--niter 50 --niter_decay 50 --save_period 25 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/Clean-VITON/VITON_traindata \
--valroot ../dataset/Clean-VITON/VITON_test \
--pb_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_e2e_100/weights/pb_warp_last.pt' \
--pb_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_e2e_100/weights/pb_gen_last.pt' \
# --pf_warp_checkpoint 'runs/train/PFAFN_align_mobile_merge-viton-v2/PFAFN_PF_warp_align_mobile_100/weights/pf_warp_last.pt' 