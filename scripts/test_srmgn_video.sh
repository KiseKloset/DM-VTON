name="PFAFN_PF_e2e_align_mobile_100"
type="093"

python test_srmgn_pf_video.py \
    --name srmgn_"$type"_1 \
    --gpu_ids 1 \
    --align_corners \
    --batchSize 1 --resize_or_crop None \
    --dataroot ../dataset/Flow-Style-VTON/VITON_test \
    --warp_checkpoint checkpoints/"$name"/PFAFN_warp_epoch_"$type".pth \
    --gen_checkpoint checkpoints/"$name"/PFAFN_gen_epoch_"$type".pth \
    --is_video \
    --input_video "demo.mp4" \
    --target_clothes "000048_1.jpg"
