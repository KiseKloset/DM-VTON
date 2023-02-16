python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 \
train_PBAFN_e2e_fs.py --name PBAFN_e2e_fs \
--gpu_ids 0 \
--launcher pytorch  --resize_or_crop None \
--batchSize 32  --label_nc 14 \
--niter 50 --niter_decay 150 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_stage1_fs/PBAFN_warp_epoch_201.pth' \
--checkpoints_dir checkpoints_fs \
&>> ./output/train_PBAFN_e2e_fs.txt