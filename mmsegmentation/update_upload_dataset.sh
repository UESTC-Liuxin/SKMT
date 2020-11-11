#切换环境
source activate mmlab

#合并数据集
root = '/media/Program/CV/Project/SKMT/mmsegmentation'
python /media/Program/CV/Project/SKMT/mmsegmentation/tools/convert_datasets/contact_coco.py

#随机切分
python /media/Program/CV/Project/SKMT/mmsegmentation/tools/convert_datasets/random_split.py

#上传

scp -r /media/Program/CV/dataset/SKMT/Seg liuxin@192.168.1.110:/home/liuxin/Documents/SKMT/mmsegmentation/data/SKMT/

