python test.py \
--name test_rmgn \
--gpu_ids 1 \
--batchSize 100 --resize_or_crop None \
--hr --predmask \
--dataroot ../dataset/Flow-Style-VTON/VITON_test \
--warp_checkpoint checkpoints/RMGN_warp_epoch_030.pth \
--gen_checkpoint checkpoints/RMGN_gen_epoch_030.pth 

# pid
# python -m pytorch_fid ../dataset/Flow-Style-VTON/VITON_test/test_img results/test_rmgn/tryon 