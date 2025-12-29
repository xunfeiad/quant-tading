//! ETL 管道

use crate::enrichment::{DataEnricher, MarketData};
use crate::scraper::{NewsScraper, ScraperFactory};
use crate::sentiment::SentimentAnalyzer;
use crate::storage::Storage;
use crate::types::{
    ETLConfig, ETLResult, EnrichedMarketData, NewsArticle, NewsSource, SentimentAnalysis,
};
use chrono::{DateTime, Duration, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;

/// ETL 管道
pub struct ETLPipeline {
    config: ETLConfig,
    scrapers: Vec<Box<dyn NewsScraper>>,
    sentiment_analyzer: SentimentAnalyzer,
    enricher: DataEnricher,
    storage: Option<Arc<RwLock<Storage>>>,
}

impl ETLPipeline {
    /// 创建新的 ETL 管道
    pub async fn new(config: ETLConfig) -> ETLResult<Self> {
        // 创建爬虫
        let scrapers: Vec<Box<dyn NewsScraper>> = config
            .enabled_sources
            .iter()
            .map(|&source| ScraperFactory::create(source))
            .collect();

        // 创建存储
        let storage = if config.enable_cache {
            let s = Storage::new(&config.database_url).await?;
            Some(Arc::new(RwLock::new(s)))
        } else {
            None
        };

        Ok(Self {
            config,
            scrapers,
            sentiment_analyzer: SentimentAnalyzer::new(),
            enricher: DataEnricher::new(),
            storage,
        })
    }

    /// 抓取新闻
    pub async fn fetch_news(&self, limit_per_source: usize) -> ETLResult<Vec<NewsArticle>> {
        tracing::info!("Fetching news from {} sources", self.scrapers.len());

        let mut all_articles = Vec::new();

        // 并发抓取多个数据源
        let futures: Vec<_> = self
            .scrapers
            .iter()
            .map(|scraper| async move {
                match scraper.fetch_latest(limit_per_source).await {
                    Ok(articles) => {
                        tracing::info!(
                            "Fetched {} articles from {}",
                            articles.len(),
                            scraper.source()
                        );
                        articles
                    }
                    Err(e) => {
                        tracing::warn!("Failed to fetch from {}: {}", scraper.source(), e);
                        Vec::new()
                    }
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        for articles in results {
            all_articles.extend(articles);
        }

        // 保存到数据库
        if let Some(storage) = &self.storage {
            let storage = storage.read().await;
            let saved = storage.save_articles(&all_articles).await?;
            tracing::info!("Saved {} articles to database", saved);
        }

        tracing::info!("Total fetched: {} articles", all_articles.len());
        Ok(all_articles)
    }

    /// 分析情感
    pub fn analyze_sentiment(&self, articles: &[NewsArticle]) -> Vec<SentimentAnalysis> {
        self.sentiment_analyzer.analyze_batch(articles)
    }

    /// 增强市场数据
    pub fn enrich_data(
        &self,
        market_data: &[MarketData],
        news_articles: &[NewsArticle],
    ) -> ETLResult<Vec<EnrichedMarketData>> {
        self.enricher
            .enrich_market_data(market_data, news_articles, self.config.time_window)
    }

    /// 完整的 ETL 流程
    pub async fn run(
        &self,
        market_data: &[MarketData],
    ) -> ETLResult<Vec<EnrichedMarketData>> {
        tracing::info!("Running ETL pipeline for {} market data points", market_data.len());

        // 1. Extract: 抓取新闻
        let news = self.fetch_news(50).await?;

        // 2. Transform & Load: 分析和增强
        let enriched = self.enrich_data(market_data, &news)?;

        tracing::info!("ETL pipeline completed");
        Ok(enriched)
    }

    /// 从数据库加载历史新闻并增强数据
    pub async fn enrich_from_cache(
        &self,
        market_data: &[MarketData],
        lookback_hours: i64,
    ) -> ETLResult<Vec<EnrichedMarketData>> {
        if let Some(storage) = &self.storage {
            let storage = storage.read().await;

            let end = Utc::now();
            let start = end - Duration::hours(lookback_hours);

            let news = storage.query_by_timerange(start, end).await?;
            tracing::info!("Loaded {} articles from cache", news.len());

            self.enrich_data(market_data, &news)
        } else {
            // 没有缓存，实时抓取
            self.run(market_data).await
        }
    }

    /// 获取存储统计信息
    pub async fn storage_stats(&self) -> ETLResult<Option<String>> {
        if let Some(storage) = &self.storage {
            let storage = storage.read().await;
            let stats = storage.stats().await?;
            Ok(Some(format!("Total articles: {}", stats.total_articles)))
        } else {
            Ok(None)
        }
    }
}

/// ETL 管道构建器
pub struct ETLPipelineBuilder {
    config: ETLConfig,
}

impl ETLPipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: ETLConfig::default(),
        }
    }

    pub fn with_sources(mut self, sources: Vec<NewsSource>) -> Self {
        self.config.enabled_sources = sources;
        self
    }

    pub fn with_time_window(mut self, minutes: i64) -> Self {
        self.config.time_window = minutes;
        self
    }

    pub fn with_database(mut self, url: String) -> Self {
        self.config.database_url = url;
        self
    }

    pub fn enable_cache(mut self, enable: bool) -> Self {
        self.config.enable_cache = enable;
        self
    }

    pub async fn build(self) -> ETLResult<ETLPipeline> {
        ETLPipeline::new(self.config).await
    }
}

impl Default for ETLPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
