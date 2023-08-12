python test.py --project runs/test --name test \
--device 0 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/Clean-VITON/VITON_test \
--pf_warp_checkpoint 'runs/train/PFAFN_align_mobile_warpKDloss_clean-viton/PFAFN_PF_e2e_align_mobile_warpKDloss_100/weights/pf_warp_best.pt' \
--pf_gen_checkpoint 'runs/train/PFAFN_align_mobile_warpKDloss_clean-viton/PFAFN_PF_e2e_align_mobile_warpKDloss_100/weights/pf_gen_best.pt' 
# --dataroot ../dataset/Clean-VITON/VITON_test \
# ./dataset/DressCode \