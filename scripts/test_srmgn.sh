# type="036" # 8.95
name="SRMGN_PF_e2e_align_100"

for type in "036" "036" "036" "036" "036" "036" "036" "036" "036" "036"
do
    python test_srmgn_pf.py \
    --name srmgn_"$type"_1 \
    --gpu_ids 1 \
    --batchSize 1 --resize_or_crop None \
    --dataroot ../dataset/Flow-Style-VTON/VITON_test \
    --warp_checkpoint checkpoints/"$name"/PFAFN_warp_epoch_"$type".pth \
    --gen_checkpoint checkpoints/"$name"/PFAFN_gen_epoch_"$type".pth \
    --align_corners \
    &>> ./output/temp.txt 

    # # pid
    # python -m pytorch_fid results/srmgn_"$type"_1/tryon ../dataset/Flow-Style-VTON/VITON_test/test_img --device cuda:1
done