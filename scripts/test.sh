python test.py --name demo \
--gpu_ids 0 \
--batchSize 1 --resize_or_crop None \
--dataroot ../../dataset/Flow-Style-VTON/VITON_test \
--warp_checkpoint 'checkpoints_fs/PFAFN_e2e/PFAFN_warp_epoch_007.pth' \
--gen_checkpoint 'checkpoints_fs/PFAFN_e2e/PFAFN_gen_epoch_007.pth' \