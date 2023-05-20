python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 train_PBAFN_e2e.py \
--name SRMGN_PB_e2e_align_merge-viton-v1_100 \
--gpu_ids 1 \
--align_corners \
--launcher pytorch  --resize_or_crop none \
--batchSize 8  --label_nc 14 \
--niter 50 --niter_decay 50 --save_epoch_freq 20 \
--dataroot ../dataset/Merge-VITON-V1/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_stage1_align_merge-viton-v1_100/PBAFN_warp_epoch_101.pth' \
--checkpoints_dir checkpoints \
&>> ./output/SRMGN_PB_e2e_align_merge-viton-v1_100.txt 