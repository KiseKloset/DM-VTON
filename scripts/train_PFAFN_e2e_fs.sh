python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 \
train_PFAFN_e2e_fs.py --name PFAFN_e2e \
--gpu_ids 2 \
--launcher pytorch --resize_or_crop None \
--batchSize 8 --label_nc 14 \
--niter 3 --niter_decay 3 --save_epoch_freq 2 \
--dataroot ../../dataset/Flow-Style-VTON/VITON_traindata \
--PFAFN_warp_checkpoint 'checkpoints_fs/PFAFN_stage1_fs/PFAFN_warp_epoch_003.pth'  \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_stage1_fs/PBAFN_warp_epoch_003.pth' \
--PBAFN_gen_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_gen_epoch_003.pth'  \
--checkpoints_dir checkpoints_fs 






