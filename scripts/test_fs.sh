type=non_aug

python test_fs.py --name fs_"$type" \
--gpu_ids 1 \
--batchSize 1 --resize_or_crop None \
--dataroot ../dataset/Flow-Style-VTON/VITON_test \
--warp_checkpoint checkpoints_fs/ckp/"$type"/PFAFN_warp_epoch_101.pth \
--gen_checkpoint checkpoints_fs/ckp/"$type"/PFAFN_gen_epoch_101.pth \


# # pid
python -m pytorch_fid results/fs_"$type"/tryon ../dataset/Flow-Style-VTON/VITON_test/test_img
