python test.py \
--name test_fs_nonaug_1 \
--gpu_ids 2 \
--batchSize 1 --resize_or_crop None \
--dataroot ../../dataset/Flow-Style-VTON/VITON_test \
--warp_checkpoint checkpoints/ckp/non_aug/PFAFN_warp_epoch_101.pth \
--gen_checkpoint checkpoints/ckp/non_aug/PFAFN_gen_epoch_101.pth 

# pid
python -m pytorch_fid ../../dataset/Flow-Style-VTON/VITON_test/test_img results/test_fs_nonaug_1/tryon 