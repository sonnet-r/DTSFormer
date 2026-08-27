import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@lru_cache(maxsize=8)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@lru_cache(maxsize=2)
def _load_solar(path: str) -> pd.DataFrame:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    dates = pd.date_range("2006-01-01", periods=len(values), freq="10min")
    columns = [f"channel_{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "date", dates)
    return frame


def _calendar_features(dates: pd.Series, cycle_length: int) -> np.ndarray:
    dates = pd.to_datetime(dates)
    if cycle_length == 7:
        cycle_index = dates.dt.weekday.to_numpy()
        minute_slot = np.zeros(len(dates), dtype=np.int64)
    elif cycle_length % 24 == 0:
        steps_per_hour = cycle_length // 24
        minute_slot = np.floor(dates.dt.minute.to_numpy() * steps_per_hour / 60).astype(
            np.int64
        )
        minute_slot = np.clip(minute_slot, 0, steps_per_hour - 1)
        cycle_index = dates.dt.hour.to_numpy() * steps_per_hour + minute_slot
    else:
        cycle_index = (dates.dt.dayofyear.to_numpy() - 1) % cycle_length
        minute_slot = np.zeros(len(dates), dtype=np.int64)

    # 最后一列保存统一的周期位置，供趋势分支构造周期查询。
    return np.column_stack(
        (
            dates.dt.month.to_numpy(),
            dates.dt.day.to_numpy(),
            dates.dt.weekday.to_numpy(),
            dates.dt.hour.to_numpy(),
            minute_slot,
            cycle_index,
        )
    ).astype(np.int64)


class ForecastDataset(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        cycle_length=24,
    ):
        if size is None:
            size = (96, 48, 96)
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.cycle_length = cycle_length
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScaler()
        self._read_data()

    def _split_borders(self, sample_count):
        train_count = int(sample_count * 0.7)
        test_count = int(sample_count * 0.2)
        validation_count = sample_count - train_count - test_count
        starts = (
            0,
            train_count - self.seq_len,
            sample_count - test_count - self.seq_len,
        )
        ends = (
            train_count,
            train_count + validation_count,
            sample_count,
        )
        return starts, ends

    def _read_frame(self):
        path = os.path.abspath(os.path.join(self.root_path, self.data_path))
        return _load_csv(path)

    def _read_data(self):
        frame = self._read_frame()
        if "date" not in frame.columns:
            raise ValueError(f"{self.data_path} must contain a 'date' column")
        if self.features in ("M", "MS"):
            feature_columns = [column for column in frame.columns if column != "date"]
            if self.features == "MS":
                if self.target not in feature_columns:
                    raise ValueError(f"Target column '{self.target}' is missing")
                feature_columns.remove(self.target)
                feature_columns.append(self.target)
        else:
            if self.target not in frame.columns:
                raise ValueError(f"Target column '{self.target}' is missing")
            feature_columns = [self.target]

        # 归一化统计量只由训练集计算，避免验证集和测试集信息泄漏。
        starts, ends = self._split_borders(len(frame))
        split_index = {"train": 0, "val": 1, "test": 2}[self.flag]
        start, end = starts[split_index], ends[split_index]
        values = frame[feature_columns].to_numpy(dtype=np.float32)
        split_values = values[start:end]
        if self.scale:
            self.scaler.fit(values[starts[0] : ends[0]])
            split_values = self.scaler.transform(split_values).astype(np.float32)

        dates = frame["date"].iloc[start:end]
        stamps = _calendar_features(dates, self.cycle_length)
        self.data_x = split_values
        self.data_y = self.data_x
        self.data_stamp = stamps

    def __getitem__(self, index):
        sequence_start = index
        sequence_end = sequence_start + self.seq_len
        target_start = sequence_end - self.label_len
        target_end = target_start + self.label_len + self.pred_len
        return (
            self.data_x[sequence_start:sequence_end],
            self.data_y[target_start:target_end],
            self.data_stamp[sequence_start:sequence_end],
            self.data_stamp[target_start:target_end],
        )

    def __len__(self):
        return max(len(self.data_x) - self.seq_len - self.pred_len + 1, 0)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class ETTHourDataset(ForecastDataset):
    def _split_borders(self, sample_count):
        del sample_count
        train_end = 12 * 30 * 24
        validation_end = train_end + 4 * 30 * 24
        test_end = validation_end + 4 * 30 * 24
        return (
            (0, train_end - self.seq_len, validation_end - self.seq_len),
            (train_end, validation_end, test_end),
        )


class ETTMinuteDataset(ForecastDataset):
    def _split_borders(self, sample_count):
        del sample_count
        train_end = 12 * 30 * 24 * 4
        validation_end = train_end + 4 * 30 * 24 * 4
        test_end = validation_end + 4 * 30 * 24 * 4
        return (
            (0, train_end - self.seq_len, validation_end - self.seq_len),
            (train_end, validation_end, test_end),
        )


class SolarDataset(ForecastDataset):
    def _read_frame(self):
        path = os.path.abspath(os.path.join(self.root_path, self.data_path))
        return _load_solar(path)
