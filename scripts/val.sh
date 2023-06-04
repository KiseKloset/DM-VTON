python val.py --project runs/test --name a \
--device 0 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/Merge-VITON-V1/VITON_test \
--pf_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100/weights/pf_warp_best.pt' \
--pf_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100/weights/pf_gen_best.pt' 