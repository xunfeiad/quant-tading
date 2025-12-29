//! RSS 通用爬虫

use super::{create_http_client, generate_article_id, NewsScraper};
use crate::types::{ETLResult, NewsArticle, NewsSource};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

pub struct RssScraper {
    client: reqwest::Client,
    source: NewsSource,
    feed_url: String,
}

impl RssScraper {
    pub fn new(source: NewsSource) -> Self {
        let feed_url = match source {
            NewsSource::Bitcoin_com => "https://news.bitcoin.com/feed/".to_string(),
            NewsSource::TheBlock => "https://www.theblock.co/rss.xml".to_string(),
            NewsSource::Decrypt => "https://decrypt.co/feed".to_string(),
            _ => String::new(),
        };

        Self {
            client: create_http_client().unwrap(),
            source,
            feed_url,
        }
    }

    pub fn with_url(source: NewsSource, feed_url: String) -> Self {
        Self {
            client: create_http_client().unwrap(),
            source,
            feed_url,
        }
    }
}

#[async_trait]
impl NewsScraper for RssScraper {
    fn source(&self) -> NewsSource {
        self.source
    }

    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        if self.feed_url.is_empty() {
            return Ok(Vec::new());
        }

        let response = self.client.get(&self.feed_url).send().await?;
        let content = response.bytes().await?;

        let feed = feed_rs::parser::parse(&content[..])
            .map_err(|e| crate::types::ETLError::HtmlParsing(e.to_string()))?;

        let articles: Vec<NewsArticle> = feed
            .entries
            .into_iter()
            .take(limit)
            .map(|entry| {
                let url = entry.links.first().map(|l| l.href.clone()).unwrap_or_default();

                let published_at = entry
                    .published
                    .or(entry.updated)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now);

                let content = entry
                    .content
                    .and_then(|c| c.body)
                    .or_else(|| entry.summary.map(|s| s.content))
                    .unwrap_or_default();

                NewsArticle {
                    id: generate_article_id(&url),
                    title: entry.title.map(|t| t.content).unwrap_or_default(),
                    content,
                    summary: entry.summary.map(|s| s.content),
                    source: self.source,
                    url,
                    published_at,
                    fetched_at: Utc::now(),
                    author: entry.authors.first().map(|a| a.name.clone()),
                    tags: entry.categories.iter().map(|c| c.term.clone()).collect(),
                    related_symbols: Vec::new(),
                }
            })
            .collect();

        tracing::info!("Fetched {} articles from RSS feed", articles.len());
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

    async fn search(&self, _query: &str, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        // RSS 不支持搜索，返回最新文章
        self.fetch_latest(limit).await
    }
}
