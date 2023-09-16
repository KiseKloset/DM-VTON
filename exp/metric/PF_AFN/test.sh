python test.py \
--name test_pfafn_1  \
--gpu_ids 0 \
--batchSize 1 --resize_or_crop None \
--dataroot ../../dataset/Flow-Style-VTON/VITON_test \
--warp_checkpoint checkpoints/PFAFN/warp_model_final.pth \
--gen_checkpoint checkpoints/PFAFN/gen_model_final.pth

# pid
python -m pytorch_fid ../../dataset/Flow-Style-VTON/VITON_test/test_img results/test_pfafn_1/tryon 