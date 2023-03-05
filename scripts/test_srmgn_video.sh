# type="036" # 8.95
name="SRMGN_PF_e2e_align_100"
type="036"

python test_srmgn_pf_video.py \
    --name srmgn_"$type"_1 \
    --gpu_ids 1 \
    --batchSize 1 --resize_or_crop None \
    --dataroot ../dataset/Flow-Style-VTON/VITON_test \
    --warp_checkpoint checkpoints/"$name"/PFAFN_warp_epoch_"$type".pth \
    --gen_checkpoint checkpoints/"$name"/PFAFN_gen_epoch_"$type".pth \
    --align_corners \
    --input_video "test.mp4" \
    --target_clothes "000066_1.jpg"
