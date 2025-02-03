_base_ = './pointrend_r50_4xb2-80k_floorplan-512x1024.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))#,
"""decode_head=[dict(type = 'FPNHead',
                    feature_strides=[4, 8, 16, 32],
                    num_classes=3),
            dict(type = 'PointHead',
                    num_classes=3)])"""
