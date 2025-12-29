//! # Quantitative Trading ML Engine
//!
//! 这个 crate 提供了完整的机器学习和深度学习功能，用于量化交易策略开发。
//!
//! ## 主要模块
//!
//! - `preprocessing`: 数据预处理和特征工程
//! - `timeseries`: 时间序列数据处理
//! - `models`: 机器学习和深度学习模型
//! - `strategy`: 交易策略生成器
//! - `backtest`: 回测引擎
//! - `evaluation`: 模型评估指标

pub mod preprocessing;
pub mod timeseries;
pub mod models;
pub mod strategy;
pub mod backtest;
pub mod evaluation;
pub mod types;

pub use types::{MLError, MLResult, ModelConfig, TradingSignal, Position};

/// ML Engine 的配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineConfig {
    /// 训练数据的时间窗口（天）
    pub training_window_days: usize,
    /// 验证集比例
    pub validation_ratio: f64,
    /// 测试集比例
    pub test_ratio: f64,
    /// 随机种子
    pub random_seed: u64,
    /// 是否启用 GPU
    pub use_gpu: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            training_window_days: 365,
            validation_ratio: 0.15,
            test_ratio: 0.15,
            random_seed: 42,
            use_gpu: false,
        }
    }
}
