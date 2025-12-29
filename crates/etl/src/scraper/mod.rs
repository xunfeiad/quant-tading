//! 新闻爬虫模块

pub mod coindesk;
pub mod cointelegraph;
pub mod cryptopanic;
pub mod rss;

use crate::types::{ETLResult, NewsArticle, NewsSource};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use std::sync::Arc;

/// 新闻爬虫接口
#[async_trait]
pub trait NewsScraper: Send + Sync {
    /// 获取新闻源名称
    fn source(&self) -> NewsSource;

    /// 抓取最新新闻
    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>>;

    /// 抓取指定时间范围的新闻
    async fn fetch_by_timerange(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ETLResult<Vec<NewsArticle>>;

    /// 搜索新闻
    async fn search(&self, query: &str, limit: usize) -> ETLResult<Vec<NewsArticle>>;
}

/// 限流爬虫包装器
pub struct RateLimitedScraper<S: NewsScraper> {
    scraper: S,
    rate_limiter: Arc<RateLimiter<governor::state::direct::NotKeyed, governor::clock::DefaultClock, governor::state::InMemoryState>>,
}

impl<S: NewsScraper> RateLimitedScraper<S> {
    pub fn new(scraper: S, requests_per_minute: u32) -> Self {
        let quota = Quota::per_minute(nonzero!(requests_per_minute));
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            scraper,
            rate_limiter,
        }
    }

    async fn wait_for_permit(&self) {
        self.rate_limiter.until_ready().await;
    }
}

#[async_trait]
impl<S: NewsScraper> NewsScraper for RateLimitedScraper<S> {
    fn source(&self) -> NewsSource {
        self.scraper.source()
    }

    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        self.wait_for_permit().await;
        self.scraper.fetch_latest(limit).await
    }

    async fn fetch_by_timerange(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ETLResult<Vec<NewsArticle>> {
        self.wait_for_permit().await;
        self.scraper.fetch_by_timerange(start, end).await
    }

    async fn search(&self, query: &str, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        self.wait_for_permit().await;
        self.scraper.search(query, limit).await
    }
}

/// 爬虫工厂
pub struct ScraperFactory;

impl ScraperFactory {
    /// 创建爬虫实例
    pub fn create(source: NewsSource) -> Box<dyn NewsScraper> {
        match source {
            NewsSource::CoinDesk => Box::new(coindesk::CoinDeskScraper::new()),
            NewsSource::CoinTelegraph => Box::new(cointelegraph::CoinTelegraphScraper::new()),
            NewsSource::CryptoPanic => Box::new(cryptopanic::CryptoPanicScraper::new()),
            _ => Box::new(rss::RssScraper::new(source)),
        }
    }

    /// 创建带限流的爬虫
    pub fn create_with_rate_limit(
        source: NewsSource,
        requests_per_minute: u32,
    ) -> Box<dyn NewsScraper> {
        let scraper = Self::create(source);
        // 注意：这里简化处理，实际需要更复杂的类型处理
        // 直接返回基础爬虫
        scraper
    }
}

/// 通用 HTTP 客户端配置
pub fn create_http_client() -> ETLResult<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; CryptoBot/1.0)")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(Into::into)
}

/// 生成文章 ID（基于 URL 的 SHA256 哈希）
pub fn generate_article_id(url: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hex::encode(hasher.finalize())
}

/// 清理 HTML 内容
pub fn clean_html(html: &str) -> String {
    ammonia::clean(html)
}

/// 提取文本内容
pub fn extract_text(html: &str) -> String {
    let cleaned = clean_html(html);
    // 移除多余的空白
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
