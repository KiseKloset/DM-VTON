python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 \
train_PBAFN_stage1_fs.py --name PBAFN_stage1_fs \
--gpu_ids 0 \
--launcher pytorch --resize_or_crop None \
--batchSize 32 --label_nc 14  \
--niter 50 --niter_decay 150 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--checkpoints_dir checkpoints_fs \
&>> ./output/train_PBAFN_stage1_fs.txt