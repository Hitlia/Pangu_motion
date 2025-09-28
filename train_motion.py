# region package
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from timm.scheduler import create_scheduler
from torch.amp import GradScaler
from tqdm import tqdm

from params import get_pangu_model_args, get_pangu_data_args
from weather_dataset import WeatherPanguData
from pangu_motion import Pangu_lite
from submodels import Evolution_Network
from loss_evolution import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# endregion

# region
CHANNELS = ["t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000", "t2m", "d2m", "u10m", "v10m", "tp"]

def compute_channel_weighting_helper():
    """
    auxiliary routine for predetermining channel weighting
    """

    # initialize empty tensor
    channel_weights = torch.ones(len(CHANNELS), dtype=torch.float32)

    for c, chn in enumerate(CHANNELS):
        if chn in ["u10m", "v10m"]:
            channel_weights[c] = 0.1
        elif chn in ["t2m", "d2m", "tp"]:
            channel_weights[c] = 1.0
        elif chn[0] in ["z", "u", "v", "t", "q"]:
            pressure_level = float(chn[1:])
            channel_weights[c] = 0.001 * pressure_level
        else:
            channel_weights[c] = 0.01

    # normalize
    channel_weights = channel_weights / torch.sum(channel_weights)

    return channel_weights

def make_grid(input):
    B, T, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)#.cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)#.cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    return grid

def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):
    B, T, C, H, W = input.size()
    # FLOW B, 2, C, H, W
    # GRID B, 2, H, W 
    output = list()
    for i in range(C):
        vgrid = grid + flow[:,:,i,:,:]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output_i = torch.nn.functional.grid_sample(input[:,:,i,:,:], vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
        output.append(output_i)
    return torch.stack(output,dim=2) #b, 1, c, h, w

def load_model(model, evo_net, optimizer=None, optim_evo=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False):
    """
    加载模型
    """
    start_epoch = 0
    start_step = 0
    min_loss = np.inf
    if path.exists():
        print("=> loading checkpoint '{}'".format(path))
        ckpt = torch.load(path, map_location="cpu")
        
        state_dict = ckpt['model']
        # 移除 'module.' 前缀（如果存在）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # 移除 'module.'
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        evo_state_dict = ckpt['evo_net']
        # 移除 'module.' 前缀（如果存在）
        new_evo_state_dict = {}
        for k, v in evo_state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # 移除 'module.'
            else:
                name = k
            new_evo_state_dict[name] = v
        evo_net.load_state_dict(new_evo_state_dict)
        
        if not only_model:
            optimizer.load_state_dict(ckpt['optimizer'])
            optim_evo.load_state_dict(ckpt['optim_evo'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            if ckpt['loss_scaler'] is not None:
                loss_scaler.load_state_dict(ckpt['loss_scaler'])
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            min_loss = ckpt["min_loss"]

    return start_epoch, start_step, min_loss

def save_model(model, evo_net, epoch=0, step=0, min_loss=0, optimizer=None, optim_evo=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False):
    """
    保存模型
    """
    print("=> saving checkpoint '{}'".format(path))
    if only_model:
        states = {
            'model': model.state_dict(),
            'evo_net': evo_net.state_dict(),
        }
    else:
        states = {
            'model': model.state_dict(),
            'evo_net': evo_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optim_evo': optim_evo.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'epoch': epoch,
            'step': step,
            'min_loss': min_loss
        }

    torch.save(states, path)

def generate_times(start_date, end_date, freq='6h'):
    return pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / dist.get_world_size()  # 总进程数
    return rt

def compute_rmse(out, tgt):
    return torch.sqrt(((out - tgt) ** 2).mean())

def pangu_autoregressive_rollout(model, x_air, x_surface, surface_mask, evo_air, evo_surface, steps, residual=False):
    """
    使用模型进行自回归预测。
    
    参数:
        model: 预测模型
        input_x (List): 包含多网格图和输入张量
        steps (int): 预测步数

    返回:
        pred_norm_list (List[Tensor]): 每一步预测结果组成的列表，每个元素形状为 [bs, 1, c, h, w]
    """
    input_seq_air      = x_air.clone()
    input_seq_surface      = x_surface.clone()
    pred_norm_list_air = []
    pred_norm_list_surface = []
    
    for k in range(steps):  
        input_seq_surface = torch.cat([input_seq_surface, evo_surface[:,k]],dim=1)
        input_seq_air = torch.cat([input_seq_air, evo_air[:,k]],dim=1)
        with torch.amp.autocast('cuda'):
            out_surface, out_air = model(input_seq_surface, surface_mask, input_seq_air)                     # [bs, 5, 13, h, w] [bs, 4, h, w]
        if residual: 
            out_air = out_air + input_seq_air
            out_surface = out_surface + input_seq_surface
        pred_norm_list_air.append(out_air)
        pred_norm_list_surface.append(out_surface)  
        # 切断梯度，使用detach
        input_seq_air = out_air#.detach().clone()
        input_seq_surface = out_surface#.detach().clone()

    # for k in range(steps):   
    #     with torch.amp.autocast('cuda'):
    #         out_surface, out_air = model(input_seq_surface, surface_mask, input_seq_air)                     # [bs, 5, 13, h, w] [bs, 4, h, w]
    #     if residual: 
    #         out_air = out_air + input_seq_air
    #         out_surface = out_surface + input_seq_surface
    #     pred_norm_list_air.append(out_air)
    #     pred_norm_list_surface.append(out_surface)  
    #     input_seq_air = out_air
    #     input_seq_surface = out_surface
    return torch.stack(pred_norm_list_air, dim=1), torch.stack(pred_norm_list_surface, dim=1)

def data_normalize(data_args):
    # x: b,t,c,h,w
    norm_dict = torch.load(data_args['root_path'] / data_args['norm_path'])
    var_mean = norm_dict['var_mean'][:70].cuda(non_blocking=True)
    var_std = norm_dict['var_std'][:70].cuda(non_blocking=True)
    var_max = norm_dict['var_max'][:70].cuda(non_blocking=True)
    var_min = norm_dict['var_min'][:70].cuda(non_blocking=True) 
    var1 = torch.zeros(70).cuda()
    var2 = torch.zeros(70).cuda()
    for i in range(70):
        if i == 69 or (i > 51 and i < 65):
            var1[i] = var_min[i]
            var2[i] = var_max[i] - var_min[i]
        else:
            var1[i] = var_mean[i]
            var2[i] = var_std[i]
    return var1, var2
            

# endregion

# region main
@torch.no_grad()
def valid_one_epoch(epoch, model, evo_net, data_args, dataloaders, surface_mask, channel_weight, var1, var2, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    score_all = torch.tensor(0.).to(device)
    evo_loss_all = torch.tensor(0.).to(device)
    
    channel_weight = channel_weight.cuda(non_blocking=True)
    var1 = var1.view(1, 1, 70, 1, 1)
    var2 = var2.view(1, 1, 70, 1, 1)
    
    surface_mask = surface_mask.cuda() # 60-0 80-140
    pbar = tqdm(dataloaders, total=len(dataloaders), desc=f"Epoch {epoch}")
    model.eval()
    for step, batch in enumerate(pbar):
        x_phys = batch['input'].cuda(non_blocking=True)             # [bs, t_in, 71, h, w]
        y_phys = batch['target'].cuda(non_blocking=True)  # [bs,t_pretrain_out, 71, h, w]

        x_norm = (x_phys - var1) / var2
        y_norm = (y_phys - var1) / var2
        
        #构建surface和air输入
        x_air = x_norm[:,0,:-5,:,:].view(x_norm.shape[0], 13, 5, x_norm.shape[3], x_norm.shape[4]).permute(0,2,1,3,4)#batch size, variables, levels, lat, lon
        x_surface = x_norm[:,0,-5:,:,:]#batch size, variables, lat, lon
        
        B,T,C,H,W = x_norm.shape
        last_frames = x_norm[:, -1:].to(device) # B, 1, C, H, W
        x_norm = x_norm.permute(0,2,1,3,4).reshape(B*C, -1, H, W)
        intensity, motion = evo_net(x_norm) #b*C,t,h,w -> B*C, T*2, H, W
        motion_ = motion.reshape(B, C, -1, H, W).reshape(B, C, data_args['t_pretrain_out'], 2, H, W).permute(0,2,3,1,4,5)
        intensity_ = intensity.reshape(B, C, -1, H, W).reshape(B, C, data_args['t_pretrain_out'], 1, H, W).permute(0,2,3,1,4,5) # b, t, 1, c, h, w

        series = []
        # series_bili = []

        sample_tensor = torch.zeros(B, 1, H, W).to(device)
        grid = make_grid(sample_tensor).to(device) # B, 2, H, W

        # 多步演化, 每帧截断梯度
        for i in range(data_args['t_pretrain_out']):
            x_t = last_frames.detach()

            # x_t_dot_bili = warp(x_t, motion_[:, i], grid, mode="bilinear", padding_mode="border")
            x_t_dot = warp(x_t, motion_[:, i], grid, mode="nearest", padding_mode="border") #b,1,c,h,w
            x_t_dot_dot = x_t_dot.detach() + intensity_[:, i] #b,1,c,h,w
            # last_frames_ = last_frames_

            last_frames = x_t_dot_dot
            series.append(x_t_dot_dot)
            # series_bili.append(x_t_dot_bili)

        evo_result = torch.cat(series, dim=1) # B, T, C, H, W
        # evo_result_bili = torch.cat(series_bili, dim=1)

        # ============ 演化网络损失及更新 ============
        loss_motion = motion_reg(motion_, y_norm) # B, T, 3, C, H, W
        loss_accum = accumulation_loss(
            pred_final=evo_result,
            pred_bili=None,#evo_result_bili,
            real=y_norm,  # [B, T_out,C, H, W]
        )
        loss_evo = loss_accum + 0.01 * loss_motion
        evo_loss_all += loss_evo.item()
        
        evo_result_detach = evo_result.detach() # b, t, c, h, w
        
        evo_air = evo_result_detach[:,:,:-5,:,:].view(B, data_args['t_pretrain_out'], 13, 5, H, W).permute(0,1,3,2,4,5)#batch size, t, variables, levels, lat, lon
        evo_surface = evo_result_detach[:,:,-5:,:,:] #batch size, t, variables, lat, lon

        pred_air, pred_surface = pangu_autoregressive_rollout(model, x_air, x_surface, surface_mask, evo_air, evo_surface, data_args['t_pretrain_out'])
        # 计算加权MAE损失
        pred_norm = torch.cat([pred_air.permute(0,1,3,2,4,5).contiguous().view(y_phys.shape[0], y_phys.shape[1], -1, y_phys.shape[3], y_phys.shape[4]), pred_surface],dim=2)  # b,t,c,lat, lon
        loss = (torch.abs(pred_norm - y_norm) * channel_weight).mean()

        loss_all = loss_all + loss.item()
        count += 1
        
        pred_phy_surface = pred_norm[:,:,-5:,20:-20, 20:-20] * var2[:,:,-5:,:,:] + var1[:,:,-5:,:,:]
        y_phy_surface = y_phys[:,:,-5:,20:-20, 20:-20]
        score = compute_rmse(pred_phy_surface, y_phy_surface)
        score_all = score_all + score

        pbar.set_postfix({
                'val_loss': f"{(loss_all/count).item():.4f}",
                'val_score': f"{(score_all/count).item():.4f}",
                'evo_loss': f"{(evo_loss_all/count).item():.4f}",
            })

    return loss_all / count, score_all / count, evo_loss_all / count

def train_one_epoch(epoch, model, evo_net, data_args, dataloaders, surface_mask, optimizer, optim_evo, channel_weight, var1, var2, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    evo_loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    model.train()
    
    channel_weight = channel_weight.cuda(non_blocking=True)
    var1 = var1.view(1, 1, 70, 1, 1)
    var2 = var2.view(1, 1, 70, 1, 1)
    
    surface_mask = surface_mask.cuda() # 60-0 80-140
    pbar = tqdm(dataloaders, total=len(dataloaders), desc=f"Epoch {epoch}")
    scaler = GradScaler()
    for step, batch in enumerate(pbar):
        # x, y = [x.cuda(non_blocking=True) for x in batch]                                          # [bs, t, c, h, w]
        x_phys = batch['input'].cuda(non_blocking=True)             # [bs, t_in, 69, h, w]
        y_phys = batch['target'].cuda(non_blocking=True)  # [bs, t_pretrain_out, 69, h, w]
        
        x_norm = (x_phys - var1) / var2
        y_norm = (y_phys - var1) / var2
        
        #构建surface和air输入
        x_air = x_norm[:,0,:-5,:,:].view(x_norm.shape[0], 13, 5, x_norm.shape[3], x_norm.shape[4]).permute(0,2,1,3,4)#batch size, variables, levels, lat, lon
        x_surface = x_norm[:,0,-5:,:,:]#batch size, variables, lat, lon
        
        B,T,C,H,W = x_norm.shape
        last_frames = x_norm[:, -1:].to(device) # B, 1, C, H, W
        x_norm = x_norm.permute(0,2,1,3,4).reshape(B*C, -1, H, W)
        intensity, motion = evo_net(x_norm) #b*C,t,h,w -> B*C, T*2, H, W
        motion_ = motion.reshape(B, C, -1, H, W).reshape(B, C, data_args['t_pretrain_out'], 2, H, W).permute(0,2,3,1,4,5)
        intensity_ = intensity.reshape(B, C, -1, H, W).reshape(B, C, data_args['t_pretrain_out'], 1, H, W).permute(0,2,3,1,4,5) # b, t, 1, c, h, w

        series = []
        # series_bili = []

        sample_tensor = torch.zeros(B, 1, H, W).to(device)
        grid = make_grid(sample_tensor).to(device) # B, 2, H, W

        # 多步演化, 每帧截断梯度
        for i in range(data_args['t_pretrain_out']):
            x_t = last_frames.detach()
            
            # x_t_dot_bili = warp(x_t, motion_[:, i], grid, mode="bilinear", padding_mode="border")
            x_t_dot = warp(x_t, motion_[:, i], grid, mode="nearest", padding_mode="border") #b,1,c,h,w
            x_t_dot_dot = x_t_dot.detach() + intensity_[:, i] #b,1,c,h,w
            # last_frames_ = last_frames_

            last_frames = x_t_dot_dot
            series.append(x_t_dot_dot)
            # series_bili.append(x_t_dot_bili)

        evo_result = torch.cat(series, dim=1) # B, T, C, H, W
        # evo_result_bili = torch.cat(series_bili, dim=1)

        # ============ 演化网络损失及更新 ============
        loss_motion = motion_reg(motion_, y_norm) # B, T, 3, C, H, W
        loss_accum = accumulation_loss(
            pred_final=evo_result,
            pred_bili=None,#evo_result_bili,
            real=y_norm,  # [B, T_out,C, H, W]
        )
        loss_evo = loss_accum + 0.01 * loss_motion
        evo_loss_all += loss_evo.item()
        
        if epoch < 18:
            optim_evo.zero_grad()
            loss_evo.backward(retain_graph=True)
            optim_evo.step()
        
        evo_result_detach = evo_result.detach() # b, t, c, h, w
        
        evo_air = evo_result_detach[:,:,:-5,:,:].view(B, data_args['t_pretrain_out'], 13, 5, H, W).permute(0,1,3,2,4,5)#batch size, t, variables, levels, lat, lon
        evo_surface = evo_result_detach[:,:,-5:,:,:] #batch size, t, variables, lat, lon

        optimizer.zero_grad()

        pred_air, pred_surface = pangu_autoregressive_rollout(model, x_air, x_surface, surface_mask, evo_air, evo_surface, data_args['t_pretrain_out'])
        # 计算加权MAE损失
        pred_norm = torch.cat([pred_air.permute(0,1,3,2,4,5).contiguous().view(y_phys.shape[0], y_phys.shape[1], -1, y_phys.shape[3], y_phys.shape[4]), pred_surface],dim=2)  # b,t,c,lat, lon
        loss = (torch.abs(pred_norm - y_norm) * channel_weight).mean()

        loss_all = loss_all + reduce_tensor(loss).item()  # 有多个进程，把进程0和1的loss加起来平均
        count += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix({
                'train_loss': f"{(loss_all/count).item():.4f}",
                'evo_loss': f"{(evo_loss_all/count).item():.4f}",
            })

    return loss_all / count, evo_loss_all / count

def train(local_rank, model_args, data_args, proj):
    '''
    Args:
        local_rank: 本地进程编号
        rank: 进程的global编号
        local_size: 每个节点几个进程
        model_args, data_args: 配置参数
        word_size: 进程总数
    '''
    try:
        rank = local_rank
        gpu = local_rank
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:30000',
                                rank=rank, world_size=model_args.world_size) 

        # Path
        log_path = data_args['root_path'] / data_args['train_log_path']
        latest_model_path = data_args['root_path'] / data_args['latest_model_path']
        train_times = generate_times(data_args['train_start_datetime'], data_args['train_end_datetime'])
        valid_times = generate_times(data_args['valid_start_datetime'], data_args['valid_end_datetime'])

        # Model
        model = Pangu_lite().cuda()
        evo_net = Evolution_Network(n_channels=1, n_classes=data_args['t_pretrain_out']).cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=model_args.lr, betas=(0.9, 0.95),
                                    weight_decay=model_args.weight_decay)
        optim_evo = torch.optim.Adam(evo_net.parameters(), lr=1e-3, betas=(0.5,0.999))
        lr_scheduler, _ = create_scheduler(model_args, optimizer)

        start_epoch, start_step, min_loss = load_model(model, evo_net, optimizer=optimizer, optim_evo=optim_evo, lr_scheduler=lr_scheduler, path=latest_model_path)
        if torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[gpu])

        # Dataloader
        train_dataset = WeatherPanguData(train_times, data_args['npy_path'], data_args['tp6hr_path'], input_window_size=data_args['t_in'], output_window_size=data_args['t_pretrain_out'])
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset, batch_size=model_args.batch_size,
                                sampler=train_sampler,
                                drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
        valid_dataset = WeatherPanguData(valid_times, data_args['npy_path'], data_args['tp6hr_path'], input_window_size=data_args['t_in'], output_window_size=data_args['t_pretrain_out'])
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=model_args.batch_size,
                                drop_last=True, num_workers=4, pin_memory=True)
        
        land_mask_all = np.load("./constant_mask/land_mask.npy")
        land_mask = land_mask_all[360+241:360:-1,320+720:561+720].copy()
        land_mask = torch.FloatTensor(land_mask)
        soil_type_all = np.load("./constant_mask/soil_type.npy")
        soil_type = soil_type_all[360+241:360:-1,320+720:561+720].copy()
        soil_type = torch.FloatTensor(soil_type)
        topography_all = np.load("./constant_mask/topography.npy")
        topography = topography_all[360+241:360:-1,320+720:561+720].copy()
        topography = torch.FloatTensor(topography)
        surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)
        
        channel_weight = compute_channel_weighting_helper().view(1, 1, 70, 1, 1)
        
        var1, var2 = data_normalize(data_args)

        best_score = 1e3
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, model_args.epochs):
            train_sampler.set_epoch(epoch)
            train_loss, train_evo_loss = train_one_epoch(epoch, model, evo_net, data_args, train_loader, surface_mask, optimizer, optim_evo, channel_weight, var1, var2, device=gpu)

            lr_scheduler.step(epoch)
            if gpu == 0:
                val_loss, val_score, val_evo_loss = valid_one_epoch(epoch, model, evo_net, data_args, valid_loader, surface_mask, channel_weight, var1, var2, device=gpu)

                print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train loss: {train_loss.item():.6f} | Train evo loss: {train_evo_loss.item():.6f} | Val loss: {val_loss.item():.6f}, Val score: {val_score.item():.6f} | Val evo loss: {val_evo_loss.item():.6f}", flush=True)

                save_model(model, evo_net, epoch=epoch + 1, min_loss=min_loss, optimizer=optimizer, optim_evo=optim_evo, lr_scheduler=lr_scheduler, path=latest_model_path)
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train loss: {train_loss.item():.6f} | Train evo loss: {train_evo_loss.item():.6f} | Val loss: {val_loss.item():.6f}, Val score: {val_score.item():.6f} | Val evo loss: {val_evo_loss.item():.6f}\n")

                if val_score < best_score:
                    best_score = val_score
                    min_loss = val_loss
                    save_model(model, evo_net, epoch=epoch, min_loss=min_loss, path=data_args['root_path'] / f"output/{proj}/ckpts/epoch{epoch + 1}_{val_score:.6f}_best.pt", only_model=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()



if __name__ == '__main__':

    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True
    model_args = get_pangu_model_args()
    proj='pangu_motion_0927'
    data_args = get_pangu_data_args('pangu_motion_0927')
    index = 1
    proj_base = 'pangu_motion_0927'
    while os.path.exists(str(data_args['root_path']) + "/output/" + proj):
        proj = proj_base +"_" +str(index)
        index += 1
        # import shutil
        # shutil.rmtree(str(data_args['root_path']) + "/output/" + proj)
    ckpt_path = str(data_args['root_path']) + "/output/" + proj + "/ckpts"
    log_path = str(data_args['root_path']) + "/output/" + proj + "/logs"
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    
    mp.spawn(train, args=(model_args, data_args, proj), nprocs=1, join=True)

# endregion 