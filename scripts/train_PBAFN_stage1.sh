python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 train_PBAFN_stage1.py \
--name SRMGN_PB_stage1_align_200 \
--gpu_ids 2 \
--align_corners \
--launcher pytorch --resize_or_crop None \
--batchSize 64 --label_nc 14  \
--niter 50 --niter_decay 150 --save_epoch_freq 5 \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--checkpoints_dir checkpoints \
&>> ./output/SRMGN_PB_stage1_align_200.txt
# --valroot ../dataset/SPLIT-VITON/VITON_val \