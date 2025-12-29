//! CryptoPanic API 爬虫

use super::{create_http_client, generate_article_id, NewsScraper};
use crate::types::{ETLResult, NewsArticle, NewsSource};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CryptoPanicResponse {
    results: Vec<CryptoPanicArticle>,
}

#[derive(Debug, Deserialize)]
struct CryptoPanicArticle {
    title: String,
    url: String,
    published_at: String,
    source: CryptoPanicSource,
    currencies: Option<Vec<Currency>>,
}

#[derive(Debug, Deserialize)]
struct CryptoPanicSource {
    title: String,
}

#[derive(Debug, Deserialize)]
struct Currency {
    code: String,
}

pub struct CryptoPanicScraper {
    client: reqwest::Client,
    api_key: Option<String>,
}

impl CryptoPanicScraper {
    pub fn new() -> Self {
        Self {
            client: create_http_client().unwrap(),
            api_key: std::env::var("CRYPTOPANIC_API_KEY").ok(),
        }
    }

    pub fn with_api_key(api_key: String) -> Self {
        Self {
            client: create_http_client().unwrap(),
            api_key: Some(api_key),
        }
    }
}

impl Default for CryptoPanicScraper {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NewsScraper for CryptoPanicScraper {
    fn source(&self) -> NewsSource {
        NewsSource::CryptoPanic
    }

    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        let mut url = format!(
            "https://cryptopanic.com/api/v1/posts/?public=true&page_size={}",
            limit
        );

        if let Some(api_key) = &self.api_key {
            url.push_str(&format!("&auth_token={}", api_key));
        }

        let response: CryptoPanicResponse = self.client.get(&url).send().await?.json().await?;

        let articles: Vec<NewsArticle> = response
            .results
            .into_iter()
            .map(|item| {
                let published_at = DateTime::parse_from_rfc3339(&item.published_at)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now);

                let related_symbols = item
                    .currencies
                    .unwrap_or_default()
                    .into_iter()
                    .map(|c| c.code)
                    .collect();

                NewsArticle {
                    id: generate_article_id(&item.url),
                    title: item.title,
                    content: String::new(),
                    summary: None,
                    source: NewsSource::CryptoPanic,
                    url: item.url,
                    published_at,
                    fetched_at: Utc::now(),
                    author: Some(item.source.title),
                    tags: Vec::new(),
                    related_symbols,
                }
            })
            .collect();

        tracing::info!("Fetched {} articles from CryptoPanic", articles.len());
        Ok(articles)
    }

    async fn fetch_by_timerange(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ETLResult<Vec<NewsArticle>> {
        let articles = self.fetch_latest(100).await?;

        let filtered: Vec<_> = articles
            .into_iter()
            .filter(|article| article.published_at >= start && article.published_at <= end)
            .collect();

        Ok(filtered)
    }

    async fn search(&self, query: &str, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        let mut url = format!(
            "https://cryptopanic.com/api/v1/posts/?public=true&filter={}",
            query
        );

        if let Some(api_key) = &self.api_key {
            url.push_str(&format!("&auth_token={}", api_key));
        }

        url.push_str(&format!("&page_size={}", limit));

        let response: CryptoPanicResponse = self.client.get(&url).send().await?.json().await?;

        let articles: Vec<NewsArticle> = response
            .results
            .into_iter()
            .map(|item| NewsArticle {
                id: generate_article_id(&item.url),
                title: item.title,
                content: String::new(),
                summary: None,
                source: NewsSource::CryptoPanic,
                url: item.url,
                published_at: Utc::now(),
                fetched_at: Utc::now(),
                author: Some(item.source.title),
                tags: Vec::new(),
                related_symbols: Vec::new(),
            })
            .collect();

        Ok(articles)
    }
}
