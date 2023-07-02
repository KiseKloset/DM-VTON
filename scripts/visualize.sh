python visualize.py --project  runs/visualize --name dm-vton \
--device 1 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/Clean-VITON/VITON_traindata \
--valroot ../dataset/Clean-VITON/VITON_test \
--pb_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_e2e_100/weights/pb_warp_last.pt' \
--pb_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/FS_PB_e2e_100/weights/pb_gen_last.pt' \
--pf_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100/weights/pf_warp_best.pt' \
--pf_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100/weights/pf_gen_best.pt' \