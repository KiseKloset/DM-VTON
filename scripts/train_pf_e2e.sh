python train_pf_e2e.py --project runs/train/DM-VTON_demo --name Student_e2e \
--device 0 --align_corners --batch_size 6 --workers 16 --lr 0.00005 \
--niter 50 --niter_decay 50 --save_period 20 \
--print_step 200 --sample_step 1000 \
--dataroot ../dataset/VITON-Clean/VITON_traindata \
--valroot ../dataset/VITON-Clean/VITON_test \
--pb_warp_checkpoint runs/train/DM-VTON_demo/Teacher_e2e/weights/pb_warp_last.pt \
--pb_gen_checkpoint runs/train/DM-VTON_demo/Teacher_e2e/weights/pb_gen_last.pt \
--pf_warp_checkpoint runs/train/DM-VTON_demo/Student_warp/weights/pf_warp_last.pt