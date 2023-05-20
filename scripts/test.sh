python test.py --project runs/test --name a \
--device 1 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/Merge-VITON-V1/VITON_test_forward \
--pf_warp_checkpoint 'runs/SRMGN_align_merge-viton-v1/PF_e2e_100/weights/pf_warp_last.pt' \
--pf_gen_checkpoint 'runs/SRMGN_align_merge-viton-v1/PF_e2e_100/weights/pf_gen_last.pt' 