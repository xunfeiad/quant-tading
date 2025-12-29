//! # ETL - Extract, Transform, Load
//!
//! 用于加密货币新闻数据的 ETL 管道
//!
//! ## 功能
//!
//! - 从多个新闻源抓取数据
//! - 情感分析和 NLP 处理
//! - 与交易所数据时间对齐
//! - 特征提取和增强
//! - 数据缓存和持久化

pub mod types;
pub mod scraper;
pub mod sentiment;
pub mod enrichment;
pub mod storage;
pub mod pipeline;

pub use types::{
    ETLConfig, ETLError, ETLResult, EnrichedMarketData, Entity, EntityType, NewsArticle,
    NewsSource, Sentiment, SentimentAnalysis, TimeWindow,
};

/// ETL 引擎配置
#[derive(Debug, Clone)]
pub struct ETLEngine {
    config: ETLConfig,
}

impl ETLEngine {
    pub fn new(config: ETLConfig) -> Self {
        Self { config }
    }

    pub fn with_default() -> Self {
        Self {
            config: ETLConfig::default(),
        }
    }
}
