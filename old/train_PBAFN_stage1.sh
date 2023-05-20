python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 train_PBAFN_stage1.py \
--name SRMGN_PB_stage1_align_merge-viton-v1_100 \
--gpu_ids 1 \
--align_corners \
--launcher pytorch --resize_or_crop none \
--batchSize 32 --label_nc 14  \
--niter 50 --niter_decay 50 --save_epoch_freq 20 \
--dataroot ../dataset/Merge-VITON-V1/VITON_traindata \
--checkpoints_dir checkpoints \
&>> ./output/SRMGN_PB_stage1_align_merge-viton-v1_100.txt