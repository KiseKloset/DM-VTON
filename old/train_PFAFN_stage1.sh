python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 train_PFAFN_stage1.py \
--name SRMGN_PF_stage1_align_merge-viton-v1_100 \
--gpu_ids 2 \
--align_corners \
--launcher pytorch --resize_or_crop none \
--batchSize 20 --label_nc 14 --lr 0.00003 \
--niter 50 --niter_decay 50 --save_epoch_freq 5 \
--dataroot ../dataset/DressCode \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_200/PBAFN_warp_epoch_101.pth' \
--PBAFN_gen_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_200/PBAFN_gen_epoch_101.pth'  \
--checkpoints checkpoints \
&>> ./output/SRMGN_PF_stage1_align_merge-viton-v1_100.txt


