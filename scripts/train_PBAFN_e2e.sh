python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 train_PBAFN_e2e.py \
--name SRMGN_PB_e2e_align_dresscode_200 \
--gpu_ids 2 \
--align_corners \
--launcher pytorch  --resize_or_crop None \
--batchSize 40  --label_nc 14 \
--niter 50 --niter_decay 50 --save_epoch_freq 5 \
--dataroot ../dataset/DressCode \
--PBAFN_warp_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_100/PBAFN_warp_epoch_101.pth' \
--PBAFN_gen_checkpoint 'checkpoints/SRMGN_PB_e2e_align_dresscode_100/PBAFN_gen_epoch_101.pth' \
--checkpoints_dir checkpoints \
&>> ./output/SRMGN_PB_e2e_align_dresscode_200.txt 