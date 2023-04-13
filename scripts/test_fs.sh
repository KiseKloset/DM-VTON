# type="081" #flow_style
# type="086" #srmgn
# type="93" #flow_style_mobile
# 064 10.26
name="PFAFN_PF_e2e_align_mobile_dresscode_100"

for type in "075"
do
    python -q test_fs_pf.py \
    --name dresscode_"$type"_1 \
    --gpu_ids 2 \
    --align_corners \
    --batchSize 1 --resize_or_crop None \
    --dataroot ../dataset/DressCode \
    --warp_checkpoint checkpoints/"$name"/PFAFN_warp_epoch_"$type".pth \
    --gen_checkpoint checkpoints/"$name"/PFAFN_gen_epoch_"$type".pth \
    # &>> ./output/temp_fs.txt 

    # # pid
    # python -m pytorch_fid results/fs_"$type"_1/tryon ../dataset/Flow-Style-VTON/VITON_test/test_img --device cuda:2 &>> ./output/temp.txt 
done