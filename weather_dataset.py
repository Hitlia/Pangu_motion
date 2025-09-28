import os
import numpy as np
import torch
from torch.utils.data import Dataset
    
class WeatherPanguData(Dataset):
    def __init__(self, timestamps, npy_path, tp6hr_path=None, input_window_size=1, output_window_size=20):
        """
        Args:
            timestamps: 按时间顺序排列的 datetime 对象列表，间隔为6小时
            npy_path: 存放 npy 文件的根目录，比如 "npy_path"
            input_window_size: 输入窗口大小（时间步数）
            output_window_size: 目标窗口大小（时间步数）
        """
        self.timestamps = timestamps
        self.npy_path = npy_path
        self.tp6hr_path = tp6hr_path
        # self.selected_channels = list(np.arange(0,66))+[67]+[68]+[70]
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.total_window = input_window_size + output_window_size
        # 每个样本需要 total_window 个时间点，因此总样本数如下：
        self.num_samples = len(self.timestamps) - self.total_window + 1 

    def __len__(self):
        return self.num_samples

    def _get_filepath(self, timestamp):
        """
        根据时间戳构造 npy 文件路径，文件名格式：
        npy_path/YYYY/YYYY_MM_DD_HH.npy
        例如：2016年1月1日1点 --> npy_path/2016/2016_01_01_01.npy
        """ 
        year = timestamp.year
        file_name = f"{timestamp.year}_{timestamp.month:02d}_{timestamp.day:02d}_{timestamp.hour:02d}.npy"
        file_path = os.path.join(self.npy_path, str(year), file_name)
        return file_path
    
    def _get_tp6hr_filepath(self, timestamp):
        year = timestamp.year
        file_name = f"{timestamp.year}_{timestamp.month:02d}_{timestamp.day:02d}_{timestamp.hour:02d}.npy"
        file_path = os.path.join(self.tp6hr_path, str(year), file_name)
        return file_path

    def __getitem__(self, idx):
        # 选取输入窗口和目标窗口对应的时间戳序列
        input_timestamps = self.timestamps[idx: idx + self.input_window_size]
        target_timestamps = self.timestamps[idx + self.input_window_size: idx + self.total_window]

        input_list = []
        target_list = []

        if self.tp6hr_path is not None:
            for ts in input_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :, :241]                # [70, 241, 241]
                tp6hr_file = self._get_tp6hr_filepath(ts)
                tp6hr_data = np.load(tp6hr_file)[:, :, :241]
                data[69, :, :] = tp6hr_data[0, :, :]
                input_list.append(data)
            for ts in target_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :, :241]
                tp6hr_file = self._get_tp6hr_filepath(ts)
                tp6hr_data = np.load(tp6hr_file)[:, :, :241]
                data[69, :, :] = tp6hr_data[0, :, :]
                target_list.append(data)
        else:
            for ts in input_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :, :241]                # [70, 241, 241]
                input_list.append(data)
            for ts in target_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :, :241]
                target_list.append(data)

        input_array = np.stack(input_list, axis=0)              # [input_window_size, 70, 241, 241]
        input_tensor = torch.from_numpy(input_array).float()
        input_tensor = torch.nan_to_num(input_tensor)

        target_array = np.stack(target_list, axis=0)
        target_tensor = torch.from_numpy(target_array).float()
        target_tensor = torch.nan_to_num(target_tensor)     # [output_window_size, 71, 241, 241]


        batch_data = {
            'datetime': [ts.strftime('%Y-%m-%dT%H:%M:%S') for ts in input_timestamps],
            'input': input_tensor,
            'target': target_tensor,
        }
        return batch_data