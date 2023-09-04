python visualize.py --project  runs/visualize --name DM-VTON \
--device 1 --align_corners --batch_size 1 --workers 16 \
--dataroot ../dataset/VITON-Clean/VITON_traindata \
--valroot ../dataset/VITON-Clean/VITON_test \
--pb_warp_checkpoint runs/train/DM-VTON_demo/Teacher_e2e/weights/pb_warp_last.pt \
--pb_gen_checkpoint runs/train/DM-VTON_demo/Teacher_e2e/weights/pb_gen_last.pt \
--pf_warp_checkpoint runs/train/DM-VTON_demo/Student_e2e/weights/pf_warp_last.pt \
--pf_gen_checkpoint runs/train/DM-VTON_demo/Student_e2e/weights/pf_warp_last.pt