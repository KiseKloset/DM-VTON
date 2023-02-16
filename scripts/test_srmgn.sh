type="151"

python test_srmgn.py --name srmgn_"$type" \
--gpu_ids 2 \
--batchSize 1 --resize_or_crop None \
--dataroot ../dataset/Flow-Style-VTON/VITON_test \
--warp_checkpoint checkpoints_fs/PFAFN_e2e/PFAFN_warp_epoch_"$type".pth \
--gen_checkpoint checkpoints_fs/PFAFN_e2e/PFAFN_gen_epoch_"$type".pth \


# # pid
python -m pytorch_fid results/srmgn_"$type"/tryon ../dataset/Flow-Style-VTON/VITON_test/test_img