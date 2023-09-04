python test.py --project runs/test --name DM-VTON_demo \
--device 0 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/VITON-Clean/VITON_test \
--pf_warp_checkpoint checkpoints/dmvton_pf_warp.pt \
--pf_gen_checkpoint checkpoints/dmvton_pf_gen.pt