import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from params import get_pangu_model_args, get_pangu_data_args
from pangu_motion import Pangu_lite
from submodels import Evolution_Network
from weather_dataset import WeatherPanguData

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def compute_metrics(all_preds, all_labels):
    """
    all_preds, all_labels: numpy arrays of shape [N, T, C, H, W]
    返回：
      var_metrics: dict, 每个变量的 {'rmse','mae','bias'}
      t2m_acc: float, t2m 误差<=2℃ 的百分率
      precip_scores: dict, 降水的 TS, false_alarm_rate, miss_rate, accuracy
    """
    N, T, C, H, W = all_preds.shape
    # 展平成 [N*T, H, W] 方便计算
    pred_flat = all_preds.reshape(-1, C, H, W)
    label_flat = all_labels.reshape(-1, C, H, W)

    # 各变量 RMSE, MAE, Bias
    var_metrics = {}
    for c in range(C):
        diff = pred_flat[:, c] - label_flat[:, c]
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        var_metrics[c] = {'rmse': rmse, 'mae': mae, 'bias': bias}

    # 每组的索引范围
    groups = {
        't': range(0, 13),  # t: 0-12
        'u': range(13, 26),  # u: 13-25
        'v': range(26, 39),  # v: 26-38
        'z': range(39, 52),  # z: 39-51
        'q': range(52, 65),  # q: 52-64
        't2m': range(65, 66),
        'd2m': range(66, 67),
        'u10': range(67, 68),
        'v10': range(68, 69),
        'tp6hr': range(69, 70)
    }

    # 初始化结果字典
    group_metrics = {}

    for var_name, indices in groups.items():
        rmse_list = []
        mae_list = []
        bias_list = []
        for idx in indices:
            rmse_list.append(var_metrics[idx]['rmse'])
            mae_list.append(var_metrics[idx]['mae'])
            bias_list.append(var_metrics[idx]['bias'])

        group_metrics[var_name] = {
            'rmse': sum(rmse_list) / len(rmse_list),
            'mae': sum(mae_list) / len(mae_list),
            'bias': sum(bias_list) / len(bias_list)
        }

    # t2m 准确率 (|error| <= 2°C)
    idx_t2m = 65
    err_t2m = np.abs(pred_flat[:, idx_t2m] - label_flat[:, idx_t2m])
    total_pts = err_t2m.size
    correct_2deg = np.count_nonzero(err_t2m <= 2.0)
    t2m_acc = correct_2deg / total_pts

    # 降水 TS、空报率、漏报率、正确率
    idx_precip = 69
    # 事件定义：>0 视为“有降水”
    pred_event = (pred_flat[:, idx_precip] > 0.0001)
    true_event = (label_flat[:, idx_precip] > 0.0001)
    # print("pred event = ",pred_event)
    # print("true event = ",true_event)

    hits = np.count_nonzero(pred_event & true_event)
    false_alarms = np.count_nonzero(pred_event & ~true_event)
    misses = np.count_nonzero(~pred_event & true_event)
    correct_neg = np.count_nonzero(~pred_event & ~true_event)

    ts_score = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else np.nan
    false_alarm_rate = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    miss_rate = misses / (hits + misses) if (hits + misses) > 0 else np.nan
    accuracy = (hits + correct_neg) / (hits + false_alarms + misses + correct_neg)

    precip_scores = {
        'TS_score': ts_score,
        'false_alarm_rate': false_alarm_rate,
        'miss_rate': miss_rate,
        'accuracy': accuracy
    }

    return var_metrics, group_metrics, t2m_acc, precip_scores

def generate_times(start_date, end_date, freq='6h'):
    return pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()

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
        input_seq_air = out_air
        input_seq_surface = out_surface
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

@torch.no_grad()
def test_one_epoch(model, evo_net, data_args, dataloaders, surface_mask, var1, var2, device="cuda:0"):
    pbar = tqdm(dataloaders, total=len(dataloaders))

    var1 = var1.view(1, 1, 70, 1, 1)
    var2 = var2.view(1, 1, 70, 1, 1)
    
    surface_mask = surface_mask.cuda() # 60-0 80-140

    results = []
    truths = []
    
    model.eval()
    for step, batch in enumerate(pbar):
        x_phys = batch['input'].cuda(non_blocking=True)             # [bs, t_in, 70, h, w]
        y_phys = batch['target'].cuda(non_blocking=True)  # [bs,t_pretrain_out, 70, h, w]
        
        #构建surface和air输入
        x_norm = (x_phys - var1) / var2
        y_norm = (y_phys - var1) / var2
        
        x_air = x_norm[:,0,:-5,:,:].view(x_norm.shape[0], 13, 5, x_norm.shape[3], x_norm.shape[4]).permute(0,2,1,3,4)#batch size, variables, levels, lat, lon
        x_surface = x_norm[:,0,-5:,:,:]#batch size, variables, lat, lon

        B,T,C,H,W = x_norm.shape
        last_frames = x_norm[:, -1:].to(device) # B, 1, C, H, W
        x_norm = x_norm.permute(0,2,1,3,4).reshape(B*C, -1, H, W)
        intensity, motion = evo_net(x_norm) #b*C,t,h,w -> B*C, T*2, H, W
        motion_ = motion.reshape(B, C, -1, H, W).reshape(B, C, data_args['t_final_out'], 2, H, W).permute(0,2,3,1,4,5)
        intensity_ = intensity.reshape(B, C, -1, H, W).reshape(B, C, data_args['t_final_out'], 1, H, W).permute(0,2,3,1,4,5) # b, t, 1, c, h, w

        series = []
        # series_bili = []

        sample_tensor = torch.zeros(B, 1, H, W).to(device)
        grid = make_grid(sample_tensor).to(device) # B, 2, H, W

        # 多步演化, 每帧截断梯度
        for i in range(data_args['t_final_out']):
            x_t = last_frames.detach()

            # x_t_dot_bili = warp(x_t, motion_[:, i], grid, mode="bilinear", padding_mode="border")
            x_t_dot = warp(x_t, motion_[:, i], grid, mode="nearest", padding_mode="border") #b,1,c,h,w
            x_t_dot_dot = x_t_dot.detach() + intensity_[:, i] #b,1,c,h,w

            last_frames = x_t_dot_dot
            series.append(x_t_dot_dot)
            # series_bili.append(x_t_dot_bili)

        evo_result = torch.cat(series, dim=1) # B, T, C, H, W   
        evo_result_detach = evo_result.detach() # b, t, c, h, w
        
        evo_air = evo_result_detach[:,:,:-5,:,:].view(B, data_args['t_pretrain_out'], 13, 5, H, W).permute(0,1,3,2,4,5)#batch size, t, variables, levels, lat, lon
        evo_surface = evo_result_detach[:,:,-5:,:,:] #batch size, t, variables, lat, lon

        pred_air, pred_surface = pangu_autoregressive_rollout(model, x_air, x_surface, surface_mask, evo_air, evo_surface, data_args['t_final_out']) # b,t,5,13,lat,lon
        pred_norm = torch.cat([pred_air.permute(0,1,3,2,4,5).contiguous().view(y_phys.shape[0], y_phys.shape[1], -1, y_phys.shape[3], y_phys.shape[4]), pred_surface],dim=2)  # b,t,c,lat, lon
        
        # pred_norm[:,:,-1,:,:] /= 1
        pred_phys  = pred_norm * var2 + var1
        results.append(pred_phys.cpu().numpy())
        truths.append(y_phys.cpu().numpy())

    # Process
    all_preds = np.concatenate(results, axis=0)
    all_preds[..., -1, :, :] = np.clip(all_preds[..., -1, :, :], a_min=0, a_max=None)
    all_labels = np.concatenate(truths, axis=0)
    all_labels[..., -1, :, :] = np.clip(all_labels[..., -1, :, :], a_min=0, a_max=None)

    return all_preds, all_labels
    
def test(model_args, data_args):
    '''
    Args:
        model_args, data_args: 配置参数
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Model
    best_model = "output/pangu_motion/ckpts/epoch98_1.503700_best.pt"
    model_path = os.path.join(data_args['root_path'], best_model)
    model = Pangu_lite().cuda()
    evo_net = Evolution_Network(n_channels=1, n_classes=data_args['t_final_out']).cuda()
    ckpt = torch.load(model_path, map_location='cpu')
    
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
    evo_net.load_pretrained_weights(new_evo_state_dict, old_n_classes=data_args['t_pretrain_out'])
    # evo_net.load_state_dict(new_evo_state_dict)

    # Dataloader
    test_times = generate_times(data_args['test_start_datetime'], data_args['test_end_datetime'])
    test_dataset = WeatherPanguData(test_times, data_args['npy_path'], data_args['tp6hr_path'], input_window_size=data_args['t_in'], output_window_size=data_args['t_final_out'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=model_args.batch_size,
                              drop_last=True, shuffle=False, num_workers=0, pin_memory=True)
    
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
        
    var1, var2 = data_normalize(data_args)

    preds, truths = test_one_epoch(model, evo_net, data_args, test_loader, surface_mask, var1, var2)  
    
    compare_demo(preds[0,:,-5:,:,:], truths[0,:,-5:,:,:], lat_range, lon_range)

    var_metrics, group_metrics, t2m_acc, precip_scores = compute_metrics(preds[:,:4,:,:,:], truths[:,:4,:,:,:])
    print("Variables metrics (idx -> metrics):")
    for idx, m in group_metrics.items():
        print(f"  var {idx}: RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, Bias={m['bias']:.3f}")
    print(f"\nt2m(65) |error|<=2°C accuracy: {t2m_acc*100:.2f}%")
    print("\nPrecipitation scores:")
    for k, v in precip_scores.items():
        print(f"  {k}: {v:.3f}")

    return preds, truths

def compare_demo(pred, truth, lat_range, lon_range, delta=0.25):
    i0 = int((60.0 - lat_range[0]) / delta)
    i1 = int((60.0 - lat_range[1]) / delta)
    j0 = int((lon_range[0] - 80.0) / delta)
    j1 = int((lon_range[1] - 80.0) / delta)
    print(f'lat idx: {i0} → {i1}, lon idx: {j0} → {j1}')

    pred = pred[:, :, i0:(i1 + 1), j0:(j1 + 1)] # t,5,h,w
    truth = truth[:, :, i0:(i1 + 1), j0:(j1 + 1)]

    var_names = ['t2m', 'd2m', 'u10', 'v10', 'tp']
    n_vars = len(var_names)

    #指标计算
    # RMSE、MAE、Bias
    rmse = []
    mae = []
    bias = []
    for i in range(n_vars):
        diff = pred[:1, i] - truth[:1, i]
        rmse.append(np.sqrt((diff ** 2).mean()))
        mae.append(np.abs(diff).mean())
        bias.append(diff.mean())
    
    metrics = pd.DataFrame({
        'variable': var_names,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
    })
    print("RMSE, MAE and Bias for each variable:")
    print(metrics.to_string(index=False))

    # # 折线图
    # # MAE
    # spatial_mae = np.abs(pred - truth).mean(axis=(2, 3))  # [12, 5]
    
    # for i, name in enumerate(var_names):
    #     plt.figure()
    #     plt.plot(spatial_mae[:, i], label='pred')
    #     plt.title(f'{name} Spatial Mean MAE vs Time')
    #     plt.xlabel('Time step')
    #     plt.ylabel('MAE')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    
    # # Bias
    # spatial_bias = (pred - truth).mean(axis=(2, 3))  # [12, 5]
    
    # for i, name in enumerate(var_names):
    #     plt.figure()
    #     plt.plot(spatial_bias[:, i], label='pred')
    #     plt.title(f'{name} Spatial Mean Bias vs Time')
    #     plt.xlabel('Time step')
    #     plt.ylabel('Bias')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # # 空间分布图
    # precip_idx = 0
    # precip_name = 't2m'
    
    # vmin = truth[:4, precip_idx].min()
    # vmax = truth[:4, precip_idx].max()
    
    # # var
    # for t in range(4):
    #     plt.figure(figsize=(8, 4))
    #     for j, (data, name) in enumerate(zip([pred, truth],
    #                                          ['pred', 'truth'])):
    #         ax = plt.subplot(1, 2, j + 1)
    #         im = ax.imshow(data[t, precip_idx], vmin=vmin, vmax=vmax, aspect='auto')
    #         ax.set_title(f'{name} t={t + 1}')
    #         plt.colorbar(im, ax=ax)
    #     plt.tight_layout()
    #     plt.show()

    var_cmaps = {
        't2m': 'viridis',
        'd2m': 'viridis',
        'u10': 'viridis',
        'v10': 'viridis',
        'tp': 'viridis'
    }

    # 误差图颜色表（同样顺序）
    error_cmaps = {
        't2m': 'seismic',
        'd2m': 'seismic',
        'u10': 'seismic',
        'v10': 'seismic',
        'tp': 'seismic'
    }

    for precip_idx in range(5):
        precip_name = var_names[precip_idx]
        # Gif Compare
        vmin = truth[:8, precip_idx].min()
        vmax = truth[:8, precip_idx].max()
        
        vmin1 = pred[:8, precip_idx].min()
        vmax1 = pred[:8, precip_idx].max()
        
        all_error = np.stack([
            pred[:8, precip_idx] - truth[:8, precip_idx]
        ], axis=0)
        
        vmin_error = all_error.min()
        vmax_error = all_error.max()
        
        # 绘图并保存每一帧到 frames
        frames = []
        for t in range(8):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
            # pred, error, truth
            data_row1 = [pred, pred - truth, truth]
            names_row1 = ['pred', 'error', 'truth']
            vmin_row1 = [vmin1, vmin_error, vmin]
            vmax_row1 = [vmax1, vmax_error, vmax]
            cmaps_row1 = [var_cmaps[precip_name], error_cmaps[precip_name], var_cmaps[precip_name]]
        
            # 然后使用 cmap 参数：
            for j, (data, name, vmin_, vmax_, cmap) in enumerate(zip(data_row1, names_row1, vmin_row1, vmax_row1, cmaps_row1)):
                ax = axes[j]
                im = ax.imshow(data[t, precip_idx], vmin=vmin_, vmax=vmax_, cmap=cmap, aspect='auto')
                ax.set_title(f'{name}  t={t + 1}')
                plt.colorbar(im, ax=ax)
        
            plt.tight_layout()
        
            # 保存当前帧到内存
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = Image.open(buf)
            frames.append(frame.copy())
            buf.close()
            plt.close(fig)
        
        # 保存为 GIF 动图
        frames[0].save(f'./output/predict/test0927/{precip_name}_compare.gif', format='GIF', append_images=frames[1:], save_all=True, duration=800, loop=0)
        print("GIF 保存完成")

        # Gif
        precip_data = pred[:8, precip_idx]  # shape: [8, 181, 221]
        vmin = precip_data.min()
        vmax = precip_data.max()
        precip_cmap = 'Spectral'
        frames = []
        for t in range(8):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            im = ax.imshow(precip_data[t], vmin=vmin, vmax=vmax, cmap=precip_cmap, aspect='auto')

            ax.set_title(f'{precip_name}  •  t={t + 1}', fontsize=12, weight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # 保存图像到内存
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
            buf.seek(0)
            frames.append(Image.open(buf).convert('RGB'))
            plt.close(fig)

        # 保存为 gif
        frames[0].save(f'./output/predict/test0927/{precip_name}.gif', save_all=True, append_images=frames[1:], duration=600, loop=0)
        print("✅ GIF 保存成功")

if __name__ == '__main__':
    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True
    model_args      = get_pangu_model_args()
    data_args       = get_pangu_data_args('pangu_motion')
    lat_range = [50.0, 10.0]
    lon_range = [90.0, 130.0]
    # # pred = np.random.rand(10,12,70,241,241)
    # # label = np.random.rand(10,12,70,241,241)

    pred, label = test(model_args, data_args)
    print("test done!")
    
    np.save(f'./output/predict/test0927/pangu_pred_surface.npy', pred[:,:,-5:,:,:])
    np.save(f'./output/predict/test0927/pangu_label_surface.npy', label[:,:,-5:,:,:])
    print("save done!")
    
    # pred = np.load(f'./output/predict/test0922/pangu_pred_surface.npy')
    # label = np.load(f'./output/predict/test0922/pangu_label_surface.npy')

    # compare_demo(pred[20,:,-5:,:,:], label[20,:,-5:,:,:], lat_range, lon_range)

    # var_metrics, group_metrics, t2m_acc, precip_scores = compute_metrics(pred, label)
    # print("Variables metrics (idx -> metrics):")
    # for idx, m in group_metrics.items():
    #     print(f"  var {idx}: RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, Bias={m['bias']:.3f}")
    # print(f"\nt2m(65) |error|<=2°C accuracy: {t2m_acc*100:.2f}%")
    # print("\nPrecipitation scores:")
    # for k, v in precip_scores.items():
    #     print(f"  {k}: {v:.3f}")

