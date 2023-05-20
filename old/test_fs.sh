# type="081" #flow_style
# type="086" #srmgn
# type="93" #flow_style_mobile
# 064 10.26
type="101"
name="test"
dataset="Merge-VITON-V1"

python -q test_fs_pf.py \
--name $name \
--gpu_ids 2 \
--align_corners \
--batchSize 1 --resize_or_crop None \
--dataroot ../dataset/"$dataset"/VITON_test_forward \
--warp_checkpoint checkpoints/PFAFN_PF_e2e_align_newloss_mobile_100/PFAFN_warp_epoch_$type.pth \
--gen_checkpoint checkpoints/PFAFN_PF_e2e_align_newloss_mobile_100/PFAFN_gen_epoch_$type.pth \
# --warp_checkpoint runs/SRMGN_align_merge-viton-v1/PF_e2e_100/weights/pf_warp_epoch_$type.pt \
# --gen_checkpoint runs/SRMGN_align_merge-viton-v1/PF_e2e_100/weights/pf_gen_epoch_$type.pt \
# --warp_checkpoint checkpoints/PFAFN_PF_e2e_align_mobile_100/PFAFN_warp_epoch_$type.pth \
# --gen_checkpoint checkpoints/PFAFN_PF_e2e_align_mobile_100/PFAFN_gen_epoch_$type.pth \
# &>> ./output/temp_fs.txt 

# pid
python -m pytorch_fid results/"$name"/tryon ../dataset/"$dataset"/VITON_test_forward/test_img --device cuda:2
# done