python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 \
train_PFAFN_stage1_fs.py --name PFAFN_stage1_fs  \
--gpu_ids 0 \
--launcher pytorch --resize_or_crop None \
--batchSize 16 --label_nc 14 --lr 0.00003 \
--niter 50 --niter_decay 150 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_warp_epoch_201.pth' \
--PBAFN_gen_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_gen_epoch_201.pth'  \
--checkpoints_dir checkpoints_fs \
&>> ./output/train_PFAFN_stage1_fs.txt





