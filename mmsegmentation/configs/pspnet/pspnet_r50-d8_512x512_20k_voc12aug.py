# _base_ = [
#     '../_base_/models/pspnet_r50-d8.py',
#     '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_20k.py'
# ]
# model = dict(
#     decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))


_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/skmt_voc.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=16), auxiliary_head=dict(num_classes=16))