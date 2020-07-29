import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SkmtDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'C', 'GL', 'SAS', 'SAC','SUP', 'INF',
               'LHF', 'HH', 'SCB', 'SHB', 'TM', 'SUB', 'D',
               'GC', 'LHB')

    PALETTE = [[32, 112, 48], [48,112,32], [176,240,32], [240,112,32], [112,112,32],
               [240,112,160], [176,112,160], [176,240,160],
               [48,112,160], [112,240,32], [240,240,32], [112,240,160],
               [112,112,160], [176,112,32], [48,240,32], [48,240,160]]

    def __init__(self, split, **kwargs):
        super(SkmtDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
