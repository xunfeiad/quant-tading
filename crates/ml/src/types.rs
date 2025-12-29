//! 核心类型定义

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type MLResult<T> = Result<T, MLError>;

#[derive(Debug, Error)]
pub enum MLError {
    #[error("数据预处理错误: {0}")]
    Preprocessing(String),

    #[error("模型训练错误: {0}")]
    Training(String),

    #[error("模型预测错误: {0}")]
    Prediction(String),

    #[error("数据维度不匹配: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("无效的配置: {0}")]
    InvalidConfig(String),

    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("序列化错误: {0}")]
    Serialization(String),

    #[error("PyTorch 错误: {0}")]
    Torch(String),
}

/// 交易信号
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// 强烈买入
    StrongBuy,
    /// 买入
    Buy,
    /// 持有
    Hold,
    /// 卖出
    Sell,
    /// 强烈卖出
    StrongSell,
}

impl TradingSignal {
    /// 从预测值转换为交易信号
    /// 预测值范围: [-1.0, 1.0]
    pub fn from_prediction(value: f64) -> Self {
        match value {
            v if v > 0.6 => TradingSignal::StrongBuy,
            v if v > 0.2 => TradingSignal::Buy,
            v if v < -0.6 => TradingSignal::StrongSell,
            v if v < -0.2 => TradingSignal::Sell,
            _ => TradingSignal::Hold,
        }
    }

    /// 转换为数值
    pub fn to_value(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Hold => 0.0,
            TradingSignal::Sell => -0.5,
            TradingSignal::StrongSell => -1.0,
        }
    }
}

/// 持仓方向
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Position {
    Long,
    Short,
    Neutral,
}

/// 市场数据（OHLCV）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// 预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub timestamp: DateTime<Utc>,
    /// 预测的价格变化（百分比）
    pub price_change: f64,
    /// 预测的方向
    pub signal: TradingSignal,
    /// 置信度 [0, 1]
    pub confidence: f64,
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub input_features: usize,
    pub sequence_length: Option<usize>, // 用于时序模型
    pub hyperparameters: serde_json::Value,
}

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    RandomForest,
    XGBoost,
    LinearRegression,
    LSTM,
    GRU,
    Transformer,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::RandomForest => write!(f, "RandomForest"),
            ModelType::XGBoost => write!(f, "XGBoost"),
            ModelType::LinearRegression => write!(f, "LinearRegression"),
            ModelType::LSTM => write!(f, "LSTM"),
            ModelType::GRU => write!(f, "GRU"),
            ModelType::Transformer => write!(f, "Transformer"),
        }
    }
}

/// 训练结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub model_type: ModelType,
    pub train_metrics: Metrics,
    pub validation_metrics: Metrics,
    pub training_duration_secs: f64,
}

/// 评估指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// 均方误差
    pub mse: f64,
    /// 均方根误差
    pub rmse: f64,
    /// 平均绝对误差
    pub mae: f64,
    /// R² 分数
    pub r2_score: f64,
    /// 方向准确率（预测涨跌方向的准确度）
    pub direction_accuracy: f64,
}

impl Metrics {
    pub fn new(mse: f64, rmse: f64, mae: f64, r2_score: f64, direction_accuracy: f64) -> Self {
        Self {
            mse,
            rmse,
            mae,
            r2_score,
            direction_accuracy,
        }
    }
}
