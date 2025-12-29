//! 核心类型定义

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type ETLResult<T> = Result<T, ETLError>;

#[derive(Debug, Error)]
pub enum ETLError {
    #[error("HTTP 请求失败: {0}")]
    HttpRequest(#[from] reqwest::Error),

    #[error("HTML 解析失败: {0}")]
    HtmlParsing(String),

    #[error("JSON 解析失败: {0}")]
    JsonParsing(#[from] serde_json::Error),

    #[error("数据库错误: {0}")]
    Database(String),

    #[error("缓存错误: {0}")]
    Cache(String),

    #[error("数据源错误: {0}")]
    DataSource(String),

    #[error("限流错误: {0}")]
    RateLimit(String),

    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("其他错误: {0}")]
    Other(String),
}

/// 新闻文章
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsArticle {
    /// 唯一标识（根据 URL 生成的哈希）
    pub id: String,
    /// 标题
    pub title: String,
    /// 内容（清理后的纯文本）
    pub content: String,
    /// 摘要
    pub summary: Option<String>,
    /// 来源
    pub source: NewsSource,
    /// URL
    pub url: String,
    /// 发布时间
    pub published_at: DateTime<Utc>,
    /// 抓取时间
    pub fetched_at: DateTime<Utc>,
    /// 作者
    pub author: Option<String>,
    /// 标签/类别
    pub tags: Vec<String>,
    /// 相关币种
    pub related_symbols: Vec<String>,
}

/// 新闻来源
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NewsSource {
    CoinDesk,
    CoinTelegraph,
    CryptoPanic,
    Bitcoin_com,
    TheBlock,
    Decrypt,
    RSS,
    Twitter,
    Reddit,
    Custom,
}

impl std::fmt::Display for NewsSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NewsSource::CoinDesk => write!(f, "CoinDesk"),
            NewsSource::CoinTelegraph => write!(f, "CoinTelegraph"),
            NewsSource::CryptoPanic => write!(f, "CryptoPanic"),
            NewsSource::Bitcoin_com => write!(f, "Bitcoin.com"),
            NewsSource::TheBlock => write!(f, "The Block"),
            NewsSource::Decrypt => write!(f, "Decrypt"),
            NewsSource::RSS => write!(f, "RSS"),
            NewsSource::Twitter => write!(f, "Twitter"),
            NewsSource::Reddit => write!(f, "Reddit"),
            NewsSource::Custom => write!(f, "Custom"),
        }
    }
}

/// 情感分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// 情感分数 [-1.0, 1.0]，-1 极度负面，1 极度正面
    pub score: f64,
    /// 情感分类
    pub sentiment: Sentiment,
    /// 置信度 [0.0, 1.0]
    pub confidence: f64,
    /// 关键词
    pub keywords: Vec<String>,
    /// 实体识别（币种、人物、机构等）
    pub entities: Vec<Entity>,
}

/// 情感类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sentiment {
    VeryPositive,  // 0.5 ~ 1.0
    Positive,      // 0.2 ~ 0.5
    Neutral,       // -0.2 ~ 0.2
    Negative,      // -0.5 ~ -0.2
    VeryNegative,  // -1.0 ~ -0.5
}

impl Sentiment {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s > 0.5 => Sentiment::VeryPositive,
            s if s > 0.2 => Sentiment::Positive,
            s if s < -0.5 => Sentiment::VeryNegative,
            s if s < -0.2 => Sentiment::Negative,
            _ => Sentiment::Neutral,
        }
    }

    pub fn to_value(&self) -> f64 {
        match self {
            Sentiment::VeryPositive => 1.0,
            Sentiment::Positive => 0.5,
            Sentiment::Neutral => 0.0,
            Sentiment::Negative => -0.5,
            Sentiment::VeryNegative => -1.0,
        }
    }
}

/// 命名实体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Cryptocurrency,  // 加密货币
    Person,          // 人物
    Organization,    // 机构/公司
    Location,        // 地点
    Event,           // 事件
    Number,          // 数字/金额
}

/// 增强的市场数据（原始数据 + 新闻特征）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedMarketData {
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 开盘价
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,

    // === 新闻特征 ===
    /// 新闻数量（该时间段内）
    pub news_count: usize,
    /// 平均情感分数
    pub avg_sentiment: f64,
    /// 正面新闻比例
    pub positive_ratio: f64,
    /// 负面新闻比例
    pub negative_ratio: f64,
    /// 新闻强度（新闻量 × 平均情感强度）
    pub news_intensity: f64,
    /// 提及次数（该币种被提及的次数）
    pub mention_count: usize,
    /// 热度分数（综合新闻量、情感、提及等）
    pub buzz_score: f64,
}

/// 时间聚合窗口
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeWindow {
    /// 1 小时
    Hour1,
    /// 4 小时
    Hour4,
    /// 1 天
    Day1,
    /// 自定义（分钟数）
    Custom(i64),
}

impl TimeWindow {
    pub fn to_minutes(&self) -> i64 {
        match self {
            TimeWindow::Hour1 => 60,
            TimeWindow::Hour4 => 240,
            TimeWindow::Day1 => 1440,
            TimeWindow::Custom(minutes) => *minutes,
        }
    }
}

/// ETL 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ETLConfig {
    /// 启用的数据源
    pub enabled_sources: Vec<NewsSource>,
    /// 时间窗口
    pub time_window: i64, // 分钟
    /// 抓取间隔（秒）
    pub fetch_interval_secs: u64,
    /// 最大并发请求数
    pub max_concurrent_requests: usize,
    /// 请求超时（秒）
    pub request_timeout_secs: u64,
    /// 启用缓存
    pub enable_cache: bool,
    /// 数据库路径
    pub database_url: String,
    /// Redis URL（可选）
    pub redis_url: Option<String>,
}

impl Default for ETLConfig {
    fn default() -> Self {
        Self {
            enabled_sources: vec![
                NewsSource::CoinDesk,
                NewsSource::CoinTelegraph,
                NewsSource::CryptoPanic,
            ],
            time_window: 60, // 1 小时
            fetch_interval_secs: 300, // 5 分钟
            max_concurrent_requests: 5,
            request_timeout_secs: 30,
            enable_cache: true,
            database_url: "sqlite:data/etl.db".to_string(),
            redis_url: None,
        }
    }
}
