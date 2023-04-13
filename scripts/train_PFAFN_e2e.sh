python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 train_PFAFN_e2e.py \
--name PFAFN_PF_e2e_align_mobile_dresscode_100 \
--gpu_ids 2 \
--align_corners \
--launcher pytorch --resize_or_crop None \
--batchSize 12 --label_nc 14 \
--niter 50 --niter_decay 50 --save_epoch_freq 1 --lr 0.00003 \
--dataroot ../dataset/DressCode \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_200/PBAFN_warp_epoch_101.pth' \
--PBAFN_gen_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_200/PBAFN_gen_epoch_101.pth'  \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_PF_stage1_align_mobile_dresscode_100/PFAFN_warp_epoch_101.pth' \
--PFAFN_gen_checkpoint 'checkpoints/ckp/aug/PFAFN_gen_epoch_101.pth' \
--checkpoints_dir checkpoints \
&>> ./output/PFAFN_PF_e2e_align_mobile_dresscode_100.txt






