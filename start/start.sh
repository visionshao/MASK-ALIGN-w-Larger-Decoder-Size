export PYTHONIOENCODING=utf-8
export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128
# export NCCL_DEBUG=INFO

cd /apdcephfs/share_916081/visionshao/PrivacyNMT/Mask-Align
start_dir=/apdcephfs/share_916081/visionshao/PrivacyNMT/Mask-Align/start
# bidirectional mask-align
bash /apdcephfs/share_916081/visionshao/PrivacyNMT/Mask-Align/thualign/bin/train.sh -s /apdcephfs/share_916081/visionshao/PrivacyNMT/Mask-Align/thualign/configs/user/example.config |& tee $start_dir/train.log
# joint
# bash /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/bin/train.sh -s /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/configs/user/mask_align_nmt.config
# transformer
# bash /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/bin/train.sh -s /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/configs/user/transformer.config
# unidirectional mask-align
# bash /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/bin/train.sh -s /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/configs/user/uni_mask_align.config
# joint pretrain
# bash /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/bin/train.sh -s /apdcephfs/share_916081/visionshao/programs/Mask-Align/thualign/configs/user/pretrain_mask_align_nmt.config