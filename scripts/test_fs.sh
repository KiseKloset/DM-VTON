# type="071"
name="FS_PF_e2e_align_100"

for type in "071"
do
    python -q test_fs_pf.py \
    --name fs_"$type"_1 \
    --gpu_ids 2 \
    --batchSize 1 --resize_or_crop None \
    --dataroot ../dataset/Flow-Style-VTON/VITON_test \
    --warp_checkpoint checkpoints/"$name"/PFAFN_warp_epoch_"$type".pth \
    --gen_checkpoint checkpoints/"$name"/PFAFN_gen_epoch_"$type".pth \
    --align_corners \
    # &>> ./output/temp_fs.txt 

    # # pid
    # python -m pytorch_fid results/fs_"$type"_1/tryon ../dataset/Flow-Style-VTON/VITON_test/test_img --device cuda:1
done