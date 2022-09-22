import torch

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_mmdet_dataloader

import open3d as o3d
import open3d.core as o3c

from tools import builder
from utils import misc
from torchvision import transforms
from datasets import data_transforms

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

def show_pts(pts, device):
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point["positions"] = pts[:, :3]
    pcd.point["intensities"] = pts[:, -1]    
    #pcd.point["colors"] = pts[:, -1]    

    o3d.visualization.draw_geometries([pcd.to_legacy()])
    

def main():
    # load config
    device = 'cuda:1'
    npoints = 100000
    cfg = Config.fromfile("configs/_base_/datasets/nus-lidarcam.py")
    config = Config.fromfile('cfgs/pretrain_nuscenes.yaml')
    ckpt_path = "experiments/pretrain_nuscenes/cfgs/0913/ckpt_epoch_241.pth"

    # build
    dataset = build_dataset(cfg.data.train)
    base_model = builder.model_builder(config.model).cuda(device)
        
    #map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path) #, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)

    base_model.eval()  # set model to eval mode


    '''data_loaders = [
            build_mmdet_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # `num_gpus` will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed,
                runner_type=runner_type,
                persistent_workers=cfg.data.get('persistent_workers', False))
            for ds in dataset
        ]'''        
    data_loaders = build_mmdet_dataloader(dataset,
                                        1,
                                        1,
                                        num_gpus=1,
                                        dist=0,
                                        seed=0,
                                        runner_type='EpochBasedRunner',
                                        persistent_workers=False,
                                        shuffle=True
                                        )


    # data processing
    x = next(iter(data_loaders))
    
    points = x['points'].data[0]
    points = misc.pad_points(points, npoints)
    points = misc.random_select_points(points, npoints)
    points = points.cuda()
    points = train_transforms(points)
    
    rgbs = x['img'].data[0].cuda()
    print(rgbs.shape)
        
    loss, pts_loss, rgb_loss, rgb_pred = base_model(points, rgbs)
    import pdb; pdb.set_trace()
    

    # points_batch = x['points'].data[0]   # list of len B, elements shape (N_k, 4) k=1, ..., B 
    # pc_sample = points_batch[0]   # (N_k, 4)

    # shuffled_idxs = torch.randperm(pc_sample.shape[0])[:npoints]  # shape (npoints)
    # selected_points = pc_sample[shuffled_idxs, :]   # (npoints, 4)

    # neighborhood, center = base_model.group_divider(selected_points[:, :3].unsqueeze(0).cuda(device))  
    # neighborhood = neighborhood + center.unsqueeze(2)  # local to global coordinates
    # pts_selected = neighborhood[0].view(-1, 3)#.cpu().numpy()

    # _, num_group, group_size, _ = neighborhood.shape
    # group_idx = torch.ones(num_group * group_size, 1).cuda(device)
    # for gi in range(num_group):
    #     group_idx[group_size * gi:group_size * (gi + 1)] = gi
        
    # group_idx = group_idx / num_group        
    # pts_selected = torch.cat([pts_selected, group_idx], dim=-1)
    # #pts_selected = pts_selected.cpu().numpy()

    # device = o3c.Device("CPU:0")
    # dtype = o3c.float32

    # a = o3c.Tensor(pc_sample.numpy(), device=device)
    # a2 = o3c.Tensor(selected_points.numpy(), device=device)
    # a3 = o3c.Tensor(pts_selected.cpu().numpy(), device=device)

    # print(f"a.shape: {a.shape}")
    # print(f"a.strides: {a.strides}")
    # print(f"a.dtype: {a.dtype}")
    # print(f"a.device: {a.device}")
    # print(f"a.ndim: {a.ndim}")

    # # visualization
    # show_pts(a, device)
    # show_pts(a2, device)
    # show_pts(a3, device)

if __name__ == '__main__':
    main()

