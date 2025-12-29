//! CoinTelegraph 新闻爬虫

use super::{create_http_client, NewsScraper};
use crate::types::{ETLResult, NewsArticle, NewsSource};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

pub struct CoinTelegraphScraper {
    client: reqwest::Client,
}

impl CoinTelegraphScraper {
    pub fn new() -> Self {
        Self {
            client: create_http_client().unwrap(),
        }
    }
}

impl Default for CoinTelegraphScraper {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NewsScraper for CoinTelegraphScraper {
    fn source(&self) -> NewsSource {
        NewsSource::CoinTelegraph
    }

    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        tracing::info!("Fetching from CoinTelegraph (demo)");
        // 简化实现，返回空列表
        // 实际实现类似 CoinDesk
        Ok(Vec::new())
    }

    async fn fetch_by_timerange(
        &self,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> ETLResult<Vec<NewsArticle>> {
        Ok(Vec::new())
    }

    async fn search(&self, _query: &str, _limit: usize) -> ETLResult<Vec<NewsArticle>> {
        Ok(Vec::new())
    }
}
