import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
import mmcv
import numpy as np

@DATASETS.register_module()
class SkmtDataset(CustomDataset):
    """Pascal VOC dataset.
    Args:
    split (str): Split txt file for Pascal VOC.
    """
    CLASSES = ('background', 'SAS', 'LHB', 'D',
               'HH', 'SUB', 'SUP', 'GL', 'GC',
               'SCB', 'INF', 'C', 'TM', 'SHB',
               'LHT', 'SAC', 'INS')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0]]


    def __init__(self, split, **kwargs):
        super(SkmtDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, reduce_zero_label=False,**kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    # def encode_segmap(self,mask):
    #     mask =mask.astype(int)
    #     print(mask[240][180])
    #     label_mask=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.int)
    #     for ii,rgb in enumerate(np.array(SkmtDataset.PALETTE)):
    #         print(mask,rgb)
    #         print(np.all(mask==rgb,axis=-1))
    #         # label_mask[np.where(np.all(mask==rgb,axis=-1))[:2]]=ii
    #         # print(np.max(label_mask))
    #     print(label_mask)
    #     label_mask=label_mask.astype(int)
    #     return label_mask
    #重写
    # def get_gt_seg_maps(self):
    #     """Get ground truth segmentation maps for evaluation."""
    #     gt_seg_maps = []
    #     for img_info in self.img_infos:
    #         gt_seg_mask = mmcv.imread(
    #             img_info['ann']['seg_map'], flag='unchanged', backend='pillow')
    #         gt_seg_map=self.encode_segmap(gt_seg_mask)
    #         print(np.max(gt_seg_map))
    #         if self.reduce_zero_label:
    #             # avoid using underflow conversion
    #             gt_seg_map[gt_seg_map == 0] = 255
    #             gt_seg_map = gt_seg_map - 1
    #             gt_seg_map[gt_seg_map == 254] = 255
    #
    #         gt_seg_maps.append(gt_seg_map)
    #
    #     return gt_seg_maps