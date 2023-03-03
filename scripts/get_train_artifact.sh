python get_train_artifact.py \
--name SRMGN_no_align \
--gpu_ids 2 \
--batchSize 100 --resize_or_crop None \
--dataroot ../dataset/Flow-Style-VTON/VITON_traindata \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_201.pth' \
--PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_201.pth'  \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_e2e/PFAFN_warp_epoch_056.pth' \
--PFAFN_gen_checkpoint 'checkpoints/PFAFN_e2e/PFAFN_gen_epoch_056.pth' \