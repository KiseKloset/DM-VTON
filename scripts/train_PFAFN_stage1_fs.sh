python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 \
train_PFAFN_stage1_fs.py --name PFAFN_stage1  \
--gpu_ids 2 \
--launcher pytorch --resize_or_crop None \
--batchSize 8 --label_nc 14 --lr 0.00003 \
--niter 1 --niter_decay 1 --save_epoch_freq 2 \
--dataroot ../../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_stage1_fs/PBAFN_warp_epoch_003.pth' \
--PBAFN_gen_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_gen_epoch_003.pth'  \
--checkpoints_dir checkpoints_fs 


 





