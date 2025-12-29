//! 机器学习和深度学习模型模块

pub mod traditional;
pub mod deep_learning;

use crate::types::{MLResult, ModelConfig, Prediction};
use ndarray::{Array2, Array3};
use async_trait::async_trait;

/// 模型训练接口
#[async_trait]
pub trait Model: Send + Sync {
    /// 训练模型
    async fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> MLResult<()>;

    /// 预测
    async fn predict(&self, x: &Array2<f64>) -> MLResult<Array2<f64>>;

    /// 保存模型
    async fn save(&self, path: &str) -> MLResult<()>;

    /// 加载模型
    async fn load(path: &str) -> MLResult<Self>
    where
        Self: Sized;
}

/// 时间序列模型接口
#[async_trait]
pub trait TimeSeriesModel: Send + Sync {
    /// 训练时间序列模型
    async fn train_sequence(
        &mut self,
        x_train: &Array3<f64>,
        y_train: &Array2<f64>,
    ) -> MLResult<()>;

    /// 序列预测
    async fn predict_sequence(&self, x: &Array3<f64>) -> MLResult<Array2<f64>>;

    /// 保存模型
    async fn save(&self, path: &str) -> MLResult<()>;

    /// 加载模型
    async fn load(path: &str) -> MLResult<Self>
    where
        Self: Sized;
}
