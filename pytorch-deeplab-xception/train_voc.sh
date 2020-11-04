CUDA_VISIBLE_DEVICES=1,2 python train.py --backbone resnet --lr 0.007 --workers 4  --epochs 50 --batch-size 4 --gpu-ids 1,2 --checkname deeplab-resnet --eval-interval 1 --dataset pascal
