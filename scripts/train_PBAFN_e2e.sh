python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 train_PBAFN_e2e.py \
--name SRMGN_PB_e2e_align_400 \
--gpu_ids 1 \
--align_corners \
--launcher pytorch  --resize_or_crop None \
--batchSize 40  --label_nc 14 \
--niter 100 --niter_decay 300 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_stage1_align_200/PBAFN_warp_epoch_201.pth' \
--checkpoints_dir checkpoints \
&>> ./output/SRMGN_PB_e2e_align_400.txt 
# --valroot ../dataset/SPLIT-VITON/VITON_val \