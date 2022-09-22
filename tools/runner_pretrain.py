import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

## HWCHO 220825
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_mmdet_dataloader

## HWCHO 220913
import torchvision.transforms as T
import torchvision

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    ## HWCHO 220825 -> 아래 부분들 distributed training 할거면 잘 손봐줘야 함.. 
    if config.dataset.NAME == 'nuscenes':
        nus_cfg = Config.fromfile("/data/easy/mmdetection3d/configs/_base_/datasets/nus-lidarcam.py")
        dataset = [build_dataset(nus_cfg.data.train)]
        
        runner_type = 'EpochBasedRunner' if 'runner' not in nus_cfg else nus_cfg.runner['type']
        train_sampler = None
        train_dataloader = [
            build_mmdet_dataloader(
                ds,
                config.total_bs, #cfg.data.samples_per_gpu,
                args.num_workers,
                # `num_gpus` will be ignored if distributed
                num_gpus=2, #len(cfg.gpu_ids),
                dist=args.distributed,
                seed=args.seed,
                runner_type=runner_type,
                pin_memory=True,
                persistent_workers=nus_cfg.data.get('persistent_workers', False))
            for ds in dataset
        ]
        
        # val_dataset = build_dataset(nus_cfg.data.val, dict(test_mode=True))
        # test_dataloader = build_mmdet_dataloader(
        #     val_dataset,
        #     samples_per_gpu=config.total_bs, #val_samples_per_gpu,
        #     workers_per_gpu=args.num_workers, #cfg.data.workers_per_gpu,
        #     dist=args.distributed,
        #     shuffle=False)
        eval_cfg = nus_cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = runner_type != 'IterBasedRunner'
        
        '''eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')'''
        
    else:
        # (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
        #                                                         builder.dataset_builder(args, config.dataset.val)
        (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    
    # build model    
    base_model = builder.model_builder(config.model)    
            
    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # if args.resume:
    #     builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader[0])
        ## HWCHO 220830 -> dataset_name이 nuscenes인 경우 만들어줌
        #for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader[0]):  ## 좌측은 기존 loop
        
        for idx, batch_data in enumerate(train_dataloader[0]):
            
            num_iter += 1
            n_itr = epoch * n_batches + idx            
            
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'nuscenes':
                # batch_data.keys() -> dict_keys(['img_metas', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
                # batch_data['points'] 는 DataContainer 타입, .data로 접근, batch 개수만큼 길이의 list가 list로 한 번 더 감싸여 있음
                points = batch_data['points'].data[0]
                #points = [x.cuda() for x in points] 
                points = misc.pad_points(points, npoints)
                points = misc.random_select_points(points, npoints)
                points = points.cuda()
                # points = misc.random_select_points(points, npoints) #.cuda()
                #points = misc.fps(points, npoints)                   
                
                # rgb = {}
                # rgb['f'] = batch_data['img'].data[0][:, 0, :, :896].float()
                # rgb['fr'] = batch_data['img'].data[0][:, 1, :, :896].float()
                # rgb['fl'] = batch_data['img'].data[0][:, 2, :, :896].float()
                # rgb['b'] = batch_data['img'].data[0][:, 3, :, :896].float()
                # rgb['bl'] = batch_data['img'].data[0][:, 4, :, :896].float()
                # rgb['br'] = batch_data['img'].data[0][:, 5, :, :896].float()                
                rgbs = batch_data['img'].data[0].cuda()
                                
            else:
                (taxonomy_ids, model_ids, data) = batch_data
                if dataset_name == 'ShapeNet':
                    points = data.cuda()
                elif dataset_name == 'ModelNet':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)                   
                else:
                    raise NotImplementedError(f'Train phase do not support {dataset_name}')
                
            data_time.update(time.time() - batch_start_time)

            assert points.size(1) == npoints
            points = train_transforms(points)
            #loss = base_model(points, rgb)
            loss, pts_loss, rgb_loss, rgb_pred = base_model(points, rgbs)
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                pts_loss = pts_loss.mean()
                rgb_loss = rgb_loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()]) #*1000])
            else:
                losses.update([loss.item()]) #*1000])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/pts_Loss', pts_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/rgb_Loss', rgb_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s pts_loss = %.4f rgb_loss = %.4f lr = %.6f' %
                                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                ['%.4f' % l for l in losses.val()], pts_loss, rgb_loss, optimizer.param_groups[0]['lr']), logger = logger)
            if idx % 500 == 0:
                first_samples = torch.stack([img[0] for img in rgb_pred], dim=0)
                grid_pred = torchvision.utils.make_grid(first_samples, nrow=2, normalize=True)
                train_writer.add_image('pred', grid_pred, n_itr)

                grid_gt = torchvision.utils.make_grid(rgbs[0], nrow=2, normalize=True)
                train_writer.add_image('gt', grid_gt, n_itr)
                
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        # if epoch % 5 ==0 :
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    ## HWCHO 220830 -> 아래에도 dataset_name이 nuscenes인 경우 만들어줘야 함
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass