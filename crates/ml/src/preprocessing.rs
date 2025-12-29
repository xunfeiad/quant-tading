//! 数据预处理和特征工程模块

use crate::types::{MLError, MLResult, MarketData};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// 特征缩放器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl Scaler {
    /// 从训练数据拟合缩放器
    pub fn fit(data: &Array2<f64>) -> MLResult<Self> {
        if data.is_empty() {
            return Err(MLError::Preprocessing("数据为空".to_string()));
        }

        let mean = data.mean_axis(Axis(0)).ok_or_else(|| {
            MLError::Preprocessing("无法计算均值".to_string())
        })?;

        let std = data.std_axis(Axis(0), 0.0);

        Ok(Self { mean, std })
    }

    /// 标准化数据
    pub fn transform(&self, data: &Array2<f64>) -> MLResult<Array2<f64>> {
        if data.ncols() != self.mean.len() {
            return Err(MLError::DimensionMismatch {
                expected: self.mean.len(),
                actual: data.ncols(),
            });
        }

        let mut normalized = data.clone();
        for (i, mut row) in normalized.axis_iter_mut(Axis(0)).enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                let std = if self.std[j].abs() < 1e-10 { 1.0 } else { self.std[j] };
                *val = (*val - self.mean[j]) / std;
            }
        }

        Ok(normalized)
    }

    /// 拟合并转换
    pub fn fit_transform(data: &Array2<f64>) -> MLResult<(Self, Array2<f64>)> {
        let scaler = Self::fit(data)?;
        let transformed = scaler.transform(data)?;
        Ok((scaler, transformed))
    }

    /// 反标准化
    pub fn inverse_transform(&self, data: &Array2<f64>) -> MLResult<Array2<f64>> {
        if data.ncols() != self.mean.len() {
            return Err(MLError::DimensionMismatch {
                expected: self.mean.len(),
                actual: data.ncols(),
            });
        }

        let mut denormalized = data.clone();
        for mut row in denormalized.axis_iter_mut(Axis(0)) {
            for (j, val) in row.iter_mut().enumerate() {
                let std = if self.std[j].abs() < 1e-10 { 1.0 } else { self.std[j] };
                *val = (*val * std) + self.mean[j];
            }
        }

        Ok(denormalized)
    }
}

/// 特征工程器
pub struct FeatureEngine;

impl FeatureEngine {
    /// 计算技术指标特征
    pub fn compute_technical_indicators(data: &[MarketData]) -> MLResult<Array2<f64>> {
        if data.is_empty() {
            return Err(MLError::Preprocessing("数据为空".to_string()));
        }

        let n = data.len();
        // 特征: 收益率、MA、EMA、RSI、MACD、布林带、ATR、成交量变化
        let mut features = Array2::<f64>::zeros((n, 20));

        for i in 0..n {
            let mut col_idx = 0;

            // 1. 价格变化率
            if i > 0 {
                features[[i, col_idx]] = (data[i].close - data[i - 1].close) / data[i - 1].close;
            }
            col_idx += 1;

            // 2-4. 短期、中期、长期收益率
            for period in [5, 20, 60] {
                if i >= period {
                    features[[i, col_idx]] = (data[i].close - data[i - period].close) / data[i - period].close;
                }
                col_idx += 1;
            }

            // 5-7. 移动平均线 (5, 20, 60)
            for period in [5, 20, 60] {
                if i >= period - 1 {
                    let ma: f64 = data[i.saturating_sub(period - 1)..=i]
                        .iter()
                        .map(|d| d.close)
                        .sum::<f64>() / period as f64;
                    features[[i, col_idx]] = (data[i].close - ma) / ma;
                }
                col_idx += 1;
            }

            // 8-10. 成交量移动平均 (5, 20, 60)
            for period in [5, 20, 60] {
                if i >= period - 1 {
                    let vol_ma: f64 = data[i.saturating_sub(period - 1)..=i]
                        .iter()
                        .map(|d| d.volume)
                        .sum::<f64>() / period as f64;
                    if vol_ma > 0.0 {
                        features[[i, col_idx]] = (data[i].volume - vol_ma) / vol_ma;
                    }
                }
                col_idx += 1;
            }

            // 11. RSI (14日)
            if i >= 14 {
                let rsi = Self::calculate_rsi(&data[i - 14..=i]);
                features[[i, col_idx]] = (rsi - 50.0) / 50.0; // 归一化到 [-1, 1]
            }
            col_idx += 1;

            // 12-13. MACD
            if i >= 26 {
                let (macd, signal) = Self::calculate_macd(&data[0..=i]);
                features[[i, col_idx]] = macd / data[i].close;
                col_idx += 1;
                features[[i, col_idx]] = signal / data[i].close;
                col_idx += 1;
            } else {
                col_idx += 2;
            }

            // 14-16. 布林带
            if i >= 20 {
                let (upper, middle, lower) = Self::calculate_bollinger_bands(&data[i - 19..=i]);
                features[[i, col_idx]] = (data[i].close - middle) / middle;
                col_idx += 1;
                features[[i, col_idx]] = (upper - middle) / middle;
                col_idx += 1;
                features[[i, col_idx]] = (middle - lower) / middle;
                col_idx += 1;
            } else {
                col_idx += 3;
            }

            // 17. ATR (平均真实范围)
            if i >= 14 {
                let atr = Self::calculate_atr(&data[i - 14..=i]);
                features[[i, col_idx]] = atr / data[i].close;
            }
            col_idx += 1;

            // 18-19. 价格位置
            if i > 0 {
                let high_low_range = data[i].high - data[i].low;
                if high_low_range > 0.0 {
                    features[[i, col_idx]] = (data[i].close - data[i].low) / high_low_range;
                }
                col_idx += 1;

                features[[i, col_idx]] = (data[i].high - data[i].low) / data[i].close;
                col_idx += 1;
            }

            // 20. 成交量变化
            if i > 0 && data[i - 1].volume > 0.0 {
                features[[i, 19]] = (data[i].volume - data[i - 1].volume) / data[i - 1].volume;
            }
        }

        Ok(features)
    }

    /// 计算 RSI
    fn calculate_rsi(data: &[MarketData]) -> f64 {
        if data.len() < 2 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..data.len() {
            let change = data[i].close - data[i - 1].close;
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let avg_gain = gains / (data.len() - 1) as f64;
        let avg_loss = losses / (data.len() - 1) as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// 计算 MACD
    fn calculate_macd(data: &[MarketData]) -> (f64, f64) {
        if data.len() < 26 {
            return (0.0, 0.0);
        }

        let ema12 = Self::calculate_ema(data, 12);
        let ema26 = Self::calculate_ema(data, 26);
        let macd = ema12 - ema26;

        // 简化版信号线（9日EMA）
        let signal = macd * 0.2; // 简化计算

        (macd, signal)
    }

    /// 计算 EMA
    fn calculate_ema(data: &[MarketData], period: usize) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = data[0].close;

        for item in data.iter().skip(1) {
            ema = (item.close - ema) * multiplier + ema;
        }

        ema
    }

    /// 计算布林带
    fn calculate_bollinger_bands(data: &[MarketData]) -> (f64, f64, f64) {
        if data.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let prices: Vec<f64> = data.iter().map(|d| d.close).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;

        let variance = prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        let upper = mean + 2.0 * std_dev;
        let lower = mean - 2.0 * std_dev;

        (upper, mean, lower)
    }

    /// 计算 ATR (平均真实范围)
    fn calculate_atr(data: &[MarketData]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mut tr_sum = 0.0;
        for i in 1..data.len() {
            let high_low = data[i].high - data[i].low;
            let high_close = (data[i].high - data[i - 1].close).abs();
            let low_close = (data[i].low - data[i - 1].close).abs();

            let tr = high_low.max(high_close).max(low_close);
            tr_sum += tr;
        }

        tr_sum / (data.len() - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaler() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (scaler, transformed) = Scaler::fit_transform(&data).unwrap();

        assert!(transformed.mean_axis(Axis(0)).unwrap().iter().all(|&x| x.abs() < 1e-10));

        let reconstructed = scaler.inverse_transform(&transformed).unwrap();
        assert!((reconstructed - data).mapv(|x| x.abs()).sum() < 1e-10);
    }
}
