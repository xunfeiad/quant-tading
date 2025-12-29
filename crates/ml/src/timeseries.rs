//! 时间序列数据处理模块

use crate::types::{MLError, MLResult};
use ndarray::{Array2, Array3, s};

/// 时间序列数据集构建器
pub struct TimeSeriesBuilder {
    sequence_length: usize,
    prediction_horizon: usize,
}

impl TimeSeriesBuilder {
    /// 创建新的时间序列构建器
    ///
    /// # 参数
    /// - `sequence_length`: 输入序列长度（用多少历史数据预测）
    /// - `prediction_horizon`: 预测窗口（预测未来多少步）
    pub fn new(sequence_length: usize, prediction_horizon: usize) -> Self {
        Self {
            sequence_length,
            prediction_horizon,
        }
    }

    /// 将二维特征数据转换为时间序列数据
    ///
    /// # 返回
    /// - X: (样本数, 序列长度, 特征数)
    /// - y: (样本数, 预测目标数)
    pub fn build_sequences(
        &self,
        features: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> MLResult<(Array3<f64>, Array2<f64>)> {
        if features.nrows() != targets.nrows() {
            return Err(MLError::DimensionMismatch {
                expected: features.nrows(),
                actual: targets.nrows(),
            });
        }

        let n_samples = features.nrows();
        if n_samples < self.sequence_length + self.prediction_horizon {
            return Err(MLError::Preprocessing(
                format!("数据量不足: 需要至少 {} 个样本",
                    self.sequence_length + self.prediction_horizon)
            ));
        }

        let n_features = features.ncols();
        let n_targets = targets.ncols();
        let n_sequences = n_samples - self.sequence_length - self.prediction_horizon + 1;

        let mut x = Array3::<f64>::zeros((n_sequences, self.sequence_length, n_features));
        let mut y = Array2::<f64>::zeros((n_sequences, n_targets));

        for i in 0..n_sequences {
            // 输入序列
            for t in 0..self.sequence_length {
                for f in 0..n_features {
                    x[[i, t, f]] = features[[i + t, f]];
                }
            }

            // 目标值（预测未来第 prediction_horizon 个时间点）
            let target_idx = i + self.sequence_length + self.prediction_horizon - 1;
            for t in 0..n_targets {
                y[[i, t]] = targets[[target_idx, t]];
            }
        }

        Ok((x, y))
    }

    /// 构建滑动窗口预测序列（用于预测下一个时间点）
    pub fn build_sliding_window(
        &self,
        features: &Array2<f64>,
    ) -> MLResult<Array3<f64>> {
        let n_samples = features.nrows();
        if n_samples < self.sequence_length {
            return Err(MLError::Preprocessing(
                format!("数据量不足: 需要至少 {} 个样本", self.sequence_length)
            ));
        }

        let n_features = features.ncols();
        let n_sequences = n_samples - self.sequence_length + 1;

        let mut x = Array3::<f64>::zeros((n_sequences, self.sequence_length, n_features));

        for i in 0..n_sequences {
            for t in 0..self.sequence_length {
                for f in 0..n_features {
                    x[[i, t, f]] = features[[i + t, f]];
                }
            }
        }

        Ok(x)
    }
}

/// 时间序列数据分割器
pub struct TimeSeriesSplitter;

impl TimeSeriesSplitter {
    /// 按时间顺序分割数据集
    ///
    /// # 参数
    /// - `data`: 输入数据
    /// - `train_ratio`: 训练集比例
    /// - `val_ratio`: 验证集比例
    /// - 测试集比例 = 1 - train_ratio - val_ratio
    pub fn split<T>(
        data: &[T],
        train_ratio: f64,
        val_ratio: f64,
    ) -> MLResult<(Vec<T>, Vec<T>, Vec<T>)>
    where
        T: Clone,
    {
        if train_ratio + val_ratio >= 1.0 || train_ratio <= 0.0 || val_ratio < 0.0 {
            return Err(MLError::InvalidConfig(
                "无效的数据分割比例".to_string()
            ));
        }

        let n = data.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

        let train = data[..train_end].to_vec();
        let val = data[train_end..val_end].to_vec();
        let test = data[val_end..].to_vec();

        Ok((train, val, test))
    }

    /// 分割 Array2 数据
    pub fn split_array2(
        data: &Array2<f64>,
        train_ratio: f64,
        val_ratio: f64,
    ) -> MLResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        if train_ratio + val_ratio >= 1.0 || train_ratio <= 0.0 || val_ratio < 0.0 {
            return Err(MLError::InvalidConfig(
                "无效的数据分割比例".to_string()
            ));
        }

        let n = data.nrows();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

        let train = data.slice(s![..train_end, ..]).to_owned();
        let val = data.slice(s![train_end..val_end, ..]).to_owned();
        let test = data.slice(s![val_end.., ..]).to_owned();

        Ok((train, val, test))
    }

    /// 分割 Array3 数据（时间序列）
    pub fn split_array3(
        data: &Array3<f64>,
        train_ratio: f64,
        val_ratio: f64,
    ) -> MLResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        if train_ratio + val_ratio >= 1.0 || train_ratio <= 0.0 || val_ratio < 0.0 {
            return Err(MLError::InvalidConfig(
                "无效的数据分割比例".to_string()
            ));
        }

        let n = data.shape()[0];
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

        let train = data.slice(s![..train_end, .., ..]).to_owned();
        let val = data.slice(s![train_end..val_end, .., ..]).to_owned();
        let test = data.slice(s![val_end.., .., ..]).to_owned();

        Ok((train, val, test))
    }
}

/// 时间序列特征增强
pub struct TimeSeriesAugmentor;

impl TimeSeriesAugmentor {
    /// 添加时间特征（小时、星期几、月份等）
    pub fn add_temporal_features(
        timestamps: &[chrono::DateTime<chrono::Utc>],
    ) -> Array2<f64> {
        let n = timestamps.len();
        let mut features = Array2::<f64>::zeros((n, 6));

        for (i, ts) in timestamps.iter().enumerate() {
            // 小时 (0-23) 归一化到 [0, 1]
            features[[i, 0]] = ts.hour() as f64 / 23.0;

            // 星期几 (0-6) 归一化到 [0, 1]
            features[[i, 1]] = ts.weekday().num_days_from_monday() as f64 / 6.0;

            // 月份 (1-12) 归一化到 [0, 1]
            features[[i, 2]] = (ts.month() as f64 - 1.0) / 11.0;

            // 季度 (1-4) 归一化到 [0, 1]
            let quarter = ((ts.month() - 1) / 3 + 1) as f64;
            features[[i, 3]] = (quarter - 1.0) / 3.0;

            // 年份中的天数 (1-366) 归一化到 [0, 1]
            features[[i, 4]] = ts.ordinal() as f64 / 366.0;

            // 是否周末 (0 或 1)
            let is_weekend = ts.weekday().num_days_from_monday() >= 5;
            features[[i, 5]] = if is_weekend { 1.0 } else { 0.0 };
        }

        features
    }

    /// 计算滞后特征
    pub fn add_lag_features(
        data: &Array2<f64>,
        lags: &[usize],
    ) -> MLResult<Array2<f64>> {
        if data.is_empty() {
            return Err(MLError::Preprocessing("数据为空".to_string()));
        }

        let n_samples = data.nrows();
        let n_features = data.ncols();
        let max_lag = *lags.iter().max().unwrap_or(&0);

        if n_samples <= max_lag {
            return Err(MLError::Preprocessing(
                format!("数据量不足以创建滞后特征，最大滞后为 {}", max_lag)
            ));
        }

        // 原始特征 + 滞后特征
        let total_features = n_features + (n_features * lags.len());
        let mut result = Array2::<f64>::zeros((n_samples, total_features));

        // 复制原始特征
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = data[[i, j]];
            }
        }

        // 添加滞后特征
        let mut feature_idx = n_features;
        for &lag in lags {
            for i in lag..n_samples {
                for j in 0..n_features {
                    result[[i, feature_idx + j]] = data[[i - lag, j]];
                }
            }
            feature_idx += n_features;
        }

        Ok(result)
    }

    /// 计算滚动统计特征
    pub fn add_rolling_features(
        data: &Array2<f64>,
        window: usize,
    ) -> MLResult<Array2<f64>> {
        if data.is_empty() {
            return Err(MLError::Preprocessing("数据为空".to_string()));
        }

        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples < window {
            return Err(MLError::Preprocessing(
                format!("数据量不足以创建滚动特征，窗口大小为 {}", window)
            ));
        }

        // 每个特征添加：滚动均值、滚动标准差、滚动最大值、滚动最小值
        let total_features = n_features * 5; // 原始 + 4种滚动统计
        let mut result = Array2::<f64>::zeros((n_samples, total_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                // 原始值
                result[[i, j]] = data[[i, j]];

                if i >= window - 1 {
                    let start = i + 1 - window;
                    let window_data: Vec<f64> = (start..=i)
                        .map(|idx| data[[idx, j]])
                        .collect();

                    // 滚动均值
                    let mean = window_data.iter().sum::<f64>() / window as f64;
                    result[[i, n_features + j]] = mean;

                    // 滚动标准差
                    let variance = window_data.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>() / window as f64;
                    result[[i, n_features * 2 + j]] = variance.sqrt();

                    // 滚动最大值
                    let max = window_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    result[[i, n_features * 3 + j]] = max;

                    // 滚动最小值
                    let min = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
                    result[[i, n_features * 4 + j]] = min;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_builder() {
        let features = Array2::from_shape_vec(
            (10, 2),
            (0..20).map(|x| x as f64).collect(),
        ).unwrap();

        let targets = Array2::from_shape_vec(
            (10, 1),
            (0..10).map(|x| x as f64).collect(),
        ).unwrap();

        let builder = TimeSeriesBuilder::new(3, 1);
        let (x, y) = builder.build_sequences(&features, &targets).unwrap();

        assert_eq!(x.shape(), &[7, 3, 2]);
        assert_eq!(y.shape(), &[7, 1]);
    }

    #[test]
    fn test_time_series_split() {
        let data = Array2::from_shape_vec((100, 5), vec![1.0; 500]).unwrap();
        let (train, val, test) = TimeSeriesSplitter::split_array2(&data, 0.7, 0.15).unwrap();

        assert_eq!(train.nrows(), 70);
        assert_eq!(val.nrows(), 15);
        assert_eq!(test.nrows(), 15);
    }
}
