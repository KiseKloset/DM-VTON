python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 train_PFAFN_stage1.py \
--name PFAFN_PF_stae1_align_mobile_dresscode_100  \
--gpu_ids 2 \
--align_corners \
--launcher pytorch --resize_or_crop None \
--batchSize 20 --label_nc 14 --lr 0.00003 \
--niter 50 --niter_decay 50 --save_epoch_freq 5 \
--dataroot ../dataset/DressCode \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_200/PBAFN_warp_epoch_101.pth' \
--PBAFN_gen_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_200/PBAFN_gen_epoch_101.pth'  \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_PF_e2e_align_mobile_100/PFAFN_warp_epoch_101.pth' \
--checkpoints checkpoints \
&>> ./output/PFAFN_PF_stae1_align_mobile_dresscode_100.txt


