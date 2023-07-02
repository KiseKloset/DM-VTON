python test.py --project runs/test --name pilot-study \
--device 1 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/pilot-study \
--pf_warp_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100/weights/pf_warp_best.pt' \
--pf_gen_checkpoint 'runs/train/PFAFN_align_mobile_clean-viton/PFAFN_PF_e2e_align_mobile_100/weights/pf_gen_best.pt' 
# --dataroot ../dataset/Clean-VITON/VITON_test \
# ./dataset/DressCode \