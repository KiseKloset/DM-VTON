python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 \
train_PFAFN_e2e_fs.py --name PFAFN_e2e \
--gpu_ids 0 \
--launcher pytorch --resize_or_crop None \
--batchSize 16 --label_nc 14 \
--niter 50 --niter_decay 150 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_warp_epoch_201.pth' \
--PBAFN_gen_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_gen_epoch_201.pth'  \
--PFAFN_warp_checkpoint 'checkpoints_fs/PFAFN_stage1_fs/PFAFN_warp_epoch_201.pth' \
--checkpoints_dir checkpoints_fs \
&>> ./output/train_PFAFN_e2e_fs.txt





