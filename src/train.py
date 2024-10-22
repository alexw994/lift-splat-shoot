"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
import tqdm
import random
import shutil

from .models import compile_model_train
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info
from .utils import is_dist_avail_and_initialized, init_distributed_mode, get_rank, is_main_process

def train(version,
            dataroot='/data/nuscenes',
            nepochs=10000,

            backbone=None,
            modelf=None,
            device='cuda',
            gpuid=0,
            world_size=1,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    
    dist_conf = {'world_size': world_size,
                 'dist_url': 'env://'}
    
    init_distributed_mode(dist_conf)
    distributed = dist_conf['distributed']
    if distributed:
        gpu = dist_conf['gpu']
    else:
        if device == 'cuda':
            gpu = gpuid
        else:
            gpu = None

    seed = 42
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata', distributed=distributed)

    model, model_without_ddp = compile_model_train(grid_conf, data_aug_conf, outC=1, 
                                                    device=device, distributed=distributed, gpu=gpu)

    opt = torch.optim.Adam(model_without_ddp.parameters(), lr=lr, weight_decay=weight_decay)
    if backbone:
        print(f'** Info ** loading backbone {backbone}')
        model_without_ddp.camencode.trunk.load_state_dict(torch.load(backbone), strict=False)

    init_counter = 0
    init_epoch = 0
    if modelf:
        if not os.path.exists(modelf):
            print(f'** Warning ** {modelf} is not exist')
        else:
            ckpt = torch.load(modelf)
            print('** Info ** loading model', ckpt['model'])

            model_without_ddp.load_state_dict(torch.load(ckpt['model']))
            opt.load_state_dict(torch.load(ckpt['opt']))
            init_epoch = int(ckpt['epoch'])
            init_counter = int(ckpt['counter'])


    loss_fn = SimpleLoss(pos_weight).to(gpu)

    writer = SummaryWriter(logdir=logdir)
    val_step = 50 if version == 'mini' else 1000

    total = len(trainloader)
    print(get_rank(), f'per subdataset numbers:{total}')

    model.train()
    counter = init_counter

    for epoch in range(init_epoch, nepochs):
        if distributed:
            trainloader.batch_sampler.sampler.set_epoch(epoch)

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                if is_main_process():
                    print(f'rank:{get_rank()} epoch:{epoch} counter:{counter} prog: {(100 * batchi / total):.2f}% loss:{loss.item():.3f} {((t1 - t0)):.2f}s/it')
                    writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                if is_main_process():
                    _, _, iou = get_batch_iou(preds, binimgs)
                    writer.add_scalar('train/iou', iou, counter)
                    writer.add_scalar('train/epoch', epoch, counter)
                    writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                if is_main_process():
                    val_info = get_val_info(model, valloader, loss_fn, device)
                    print('VAL', val_info)
                    writer.add_scalar('val/loss', val_info['loss'], counter)
                    writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                if is_main_process():
                    model_without_ddp.eval()
                    mname = os.path.join(logdir, "model{}.pt".format(counter))
                    print('saving', mname)
                    torch.save(model_without_ddp.state_dict(), mname)
                    model_without_ddp.train()

                    oname = os.path.join(logdir, "opt{}.pt".format(counter))
                    print('saving', oname)
                    torch.save(opt.state_dict(), oname)

                    # ckpt
                    ckpt = {'epoch': epoch,
                            'counter':counter,
                            'model': mname,
                            'opt': oname}
                    torch.save(ckpt, os.path.join(logdir, "ckpt"))
                    
