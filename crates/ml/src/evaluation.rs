//! 模型评估模块

use crate::types::{MLResult, Metrics};
use ndarray::Array2;

/// 模型评估器
pub struct Evaluator;

impl Evaluator {
    /// 计算评估指标
    pub fn evaluate(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> MLResult<Metrics> {
        let mse = Self::mean_squared_error(y_true, y_pred);
        let rmse = mse.sqrt();
        let mae = Self::mean_absolute_error(y_true, y_pred);
        let r2 = Self::r2_score(y_true, y_pred);
        let direction_accuracy = Self::direction_accuracy(y_true, y_pred);

        Ok(Metrics::new(mse, rmse, mae, r2, direction_accuracy))
    }

    /// 均方误差 (MSE)
    pub fn mean_squared_error(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let diff = y_true - y_pred;
        let squared = diff.mapv(|x| x * x);
        squared.mean().unwrap_or(0.0)
    }

    /// 平均绝对误差 (MAE)
    pub fn mean_absolute_error(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let diff = y_true - y_pred;
        let abs_diff = diff.mapv(|x| x.abs());
        abs_diff.mean().unwrap_or(0.0)
    }

    /// R² 分数
    pub fn r2_score(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let y_mean = y_true.mean().unwrap_or(0.0);

        let ss_res: f64 = (y_true - y_pred).mapv(|x| x * x).sum();
        let ss_tot: f64 = y_true.mapv(|x| (x - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            return 0.0;
        }

        1.0 - (ss_res / ss_tot)
    }

    /// 方向准确率（预测涨跌方向的准确度）
    ///
    /// 计算预测值和真实值的符号是否一致
    pub fn direction_accuracy(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let n_samples = y_true.nrows();
        if n_samples == 0 {
            return 0.0;
        }

        let mut correct = 0;
        for i in 0..n_samples {
            let true_direction = y_true.row(i).iter().sum::<f64>() >= 0.0;
            let pred_direction = y_pred.row(i).iter().sum::<f64>() >= 0.0;

            if true_direction == pred_direction {
                correct += 1;
            }
        }

        correct as f64 / n_samples as f64
    }

    /// 夏普比率（用于回测）
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - risk_free_rate;

        let variance = returns
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        excess_return / std_dev
    }

    /// 最大回撤
    pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_drawdown = 0.0;
        let mut peak = equity_curve[0];

        for &value in equity_curve.iter().skip(1) {
            if value > peak {
                peak = value;
            }

            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// 胜率
    pub fn win_rate(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        wins as f64 / returns.len() as f64
    }

    /// 盈亏比
    pub fn profit_factor(returns: &[f64]) -> f64 {
        let profits: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

        if losses == 0.0 {
            return f64::INFINITY;
        }

        profits / losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mse() {
        let y_true = array![[1.0], [2.0], [3.0]];
        let y_pred = array![[1.1], [2.1], [2.9]];

        let mse = Evaluator::mean_squared_error(&y_true, &y_pred);
        assert!((mse - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_direction_accuracy() {
        let y_true = array![[1.0], [-1.0], [2.0], [-2.0]];
        let y_pred = array![[0.5], [-0.5], [1.5], [0.5]]; // 最后一个方向错误

        let acc = Evaluator::direction_accuracy(&y_true, &y_pred);
        assert!((acc - 0.75).abs() < 1e-6); // 3/4 = 0.75
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let sharpe = Evaluator::sharpe_ratio(&returns, 0.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0];
        let dd = Evaluator::max_drawdown(&equity);
        assert!((dd - 0.136363).abs() < 1e-5); // (110-95)/110
    }
}
