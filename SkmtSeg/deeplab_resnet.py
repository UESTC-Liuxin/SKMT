# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/1 下午10:20
"""
import os
import time
import random
import numpy as np
import argparse
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.loss import Loss
from utils.summaries import TensorboardSummary
from utils.modeltools import netParams
from utils.set_logger import get_logger

from train import Trainer
from test import Tester
from dataloader.vaihingen import VaiHinGen
from modeling.erftsn import build_erftsn


def main(args, logger, summary):
    cudnn.enabled = True  # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True  # cudnn auto-tuner to find the best algorithm to use for
    # our hardware

    seed = random.randint(1, 10000)
    logger.info('======>random seed {}'.format(seed))

    random.seed(seed)  # python random seed
    np.random.seed(seed)  # set numpy random seed
    torch.manual_seed(seed)  # set random seed for cpu

    train_set = VaiHinGen(root=args.root, split='trainl', outer_size=2 * args.image_size, centre_size=args.image_size)
    test_set = VaiHinGen(root=args.root, split='testl', outer_size=2 * args.image_size, centre_size=args.image_size)
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=True, shuffle=False, **kwargs)

    # setup optimization criterion
    criterion = Loss(args)
    logger.info('======> building network')
    # set model
    model = build_erftsn(auxiliary=args.auxiliary, trunk='deeplab', img_size=args.image_size).cuda()
    logger.info("======> computing network parameters")
    total_paramters = netParams(model)
    logger.info("the number of parameters: " + str(total_paramters))

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # setup savedir
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpus) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        model = nn.DataParallel(model).cuda()

    best_epoch = 0.
    best_overall = 0.
    best_mIoU = 0.
    best_F1 = 0.

    trainer = Trainer(args=args, dataloader=train_loader, model=model,
                      optimizer=optimizer, criterion=criterion, logger=logger, summary=summary)
    start_epoch = trainer.start_epoch

    tester = Tester(args=args, dataloader=test_loader, model=model,
                    criterion=criterion, logger=logger, summary=summary)

    writer = summary.create_summary()
    for epoch in range(start_epoch, args.max_epochs):
        trainer.train_one_epoch(epoch, writer)

        if (epoch % 100 == 0):
            score, class_iou, class_F1 = tester.test_one_epoch(epoch, writer)

            for k, v in score.items():
                print('{}: {:.5f}'.format(k, v))
                logger.info('======>{0:^18} {1:^10}'.format(k, v))

            logger.info('Now print class iou')
            for k, v in class_iou.items():
                print('{}: {:.5f}'.format(k, v))
                logger.info('======>{0:^18} {1:^10}'.format(k, v))

            logger.info('Now print class_F1')
            for k, v in class_F1.items():
                logger.info('======>{0:^18} {1:^10}'.format(k, v))

            if score["Mean IoU(8) : \t"] > best_mIoU:
                best_mIoU = score["Mean IoU(8) : \t"]

            writer.add_scalar('test/Overall_Acc', score["Overall Acc : \t"],epoch)
            if score["Overall Acc : \t"] > best_overall:
                best_overall = score["Overall Acc : \t"]
                # save model in best overall Acc
                model_file_name = args.savedir + '/model.pth'
                torch.save(model.state_dict(), model_file_name)
                best_epoch = epoch

            if score["Mean F1 : \t"] > best_F1:
                best_F1 = score["Mean F1 : \t"]

            logger.info(f"best mean IoU: {best_mIoU}")
            logger.info(f"best overall : {best_overall}")
            logger.info(f"best F1: {best_F1}")
            logger.info(f"best epoch: {best_epoch}")

            # save the model
            model_file_name = args.savedir + '/model.pth'
            state = {"epoch": epoch + 1,
                     "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "criterion": criterion.state_dict()
                     }

            logger.info('======> Now begining to save model.')
            torch.save(state, model_file_name)
            logger.info('======> Save done.')


if __name__ == '__main__':

    import timeit

    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Semantic Segmentation...')

    parser.add_argument('--model', default='erftsn', type=str)
    parser.add_argument('--auxiliary', default=None, type=str)
    parser.add_argument('--root', default='data/VAI', help='data directory')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--max_epochs', type=int, default=6500, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--show_interval', default=10, type=int)
    parser.add_argument('--show_val_interval', default=1000, type=int)
    parser.add_argument('--savedir', default="./runs/deeplabv3_resnet/",
                        help="directory to save the model snapshot")
    # parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--resume', default=None, help="the resume model path")
    args = parser.parse_args()

    # 设置运行id
    run_id = 'lr{}_bz{}'.format(args.lr, args.batch_size) \
             + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')  # 现在

    args.savedir = os.path.join(args.savedir, str(run_id))

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    logger = get_logger(args.savedir)
    logger.info('just do it')
    logger.info('Now run_id {}'.format(run_id))

    if (args.resume):
        if not os.path.exists(args.resume):
            raise Exception("the path of resume is empty!!")

    # 设置tensorboard
    summary = TensorboardSummary(args.savedir)

    logger.info('======>Input arguments:')
    for key, val in vars(args).items():
        logger.info('======> {:16} {}'.format(key, val))

    # 开始运行.........
    main(args, logger, summary)
    end = timeit.default_timer()
    logger.info("training time:", 1.0 * (end - start) / 3600)
    logger.info('model save in {}.'.format(run_id))


