# _base_ = [
#     '../_base_/models/deeplabv3_r50-d8.py',
#     '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_20k.py'
# ]
# model = dict(
#     decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))


_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/US.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))