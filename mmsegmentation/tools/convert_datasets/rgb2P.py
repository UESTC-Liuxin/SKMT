import os
import cv2
from PIL import Image

data_root='../../data/VOCdevkit/Seg'
seg_class_path=os.path.join(data_root,'skm5/SegmentationClass')

img_files=os.listdir(seg_class_path)
for file_name in img_files:
    file_path=os.path.join(seg_class_path,file_name)
    img =Image.open(file_path)
    img=img.convert('P')
    out_path=os.path.join(data_root,'skm5/SegmentationClass8bit',file_name)
    img.save(out_path)