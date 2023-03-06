python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 train_PFAFN_stage1.py \
--name SRMGN_PF_stage1_align_mobile_100 \
--gpu_ids 2 \
--launcher pytorch --resize_or_crop None \
--batchSize 20 --label_nc 14 --lr 0.00003 \
--niter 100 --niter_decay 100 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_e2e_align_200/PBAFN_warp_epoch_201.pth' \
--PBAFN_gen_checkpoint 'checkpoints/SRMGN_PB_e2e_align_200/PBAFN_gen_epoch_201.pth'  \
--checkpoints checkpoints \
&>> ./output/SRMGN_PF_stage1_align_mobile_100.txt



