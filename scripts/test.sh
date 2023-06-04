python test.py --project runs/test --name a \
--device 1 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/Clean-VITON/VITON_test \
--pf_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100-1/weights/pf_warp_best.pt' \
--pf_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100-1/weights/pf_gen_best.pt' 