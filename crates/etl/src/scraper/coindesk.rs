//! CoinDesk 新闻爬虫

use super::{create_http_client, extract_text, generate_article_id, NewsScraper};
use crate::types::{ETLResult, NewsArticle, NewsSource};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use scraper::{Html, Selector};

pub struct CoinDeskScraper {
    client: reqwest::Client,
    base_url: String,
}

impl CoinDeskScraper {
    pub fn new() -> Self {
        Self {
            client: create_http_client().unwrap(),
            base_url: "https://www.coindesk.com".to_string(),
        }
    }

    async fn fetch_page(&self, url: &str) -> ETLResult<Html> {
        let response = self.client.get(url).send().await?;
        let html = response.text().await?;
        Ok(Html::parse_document(&html))
    }

    fn parse_article_list(&self, document: &Html) -> Vec<NewsArticle> {
        let mut articles = Vec::new();

        // CoinDesk 文章选择器（需要根据实际网页结构调整）
        let article_selector = Selector::parse("article, .article-card").unwrap();
        let title_selector = Selector::parse("h2, h3, .article-title, a[data-track-link]").unwrap();
        let link_selector = Selector::parse("a").unwrap();
        let time_selector = Selector::parse("time, .timestamp").unwrap();

        for article_el in document.select(&article_selector) {
            // 提取标题
            let title = article_el
                .select(&title_selector)
                .next()
                .map(|el| el.text().collect::<String>().trim().to_string())
                .unwrap_or_default();

            // 提取链接
            let url = article_el
                .select(&link_selector)
                .next()
                .and_then(|el| el.value().attr("href"))
                .map(|href| {
                    if href.starts_with("http") {
                        href.to_string()
                    } else {
                        format!("{}{}", self.base_url, href)
                    }
                })
                .unwrap_or_default();

            // 提取时间
            let published_at = article_el
                .select(&time_selector)
                .next()
                .and_then(|el| el.value().attr("datetime"))
                .and_then(|dt| DateTime::parse_from_rfc3339(dt).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            if !title.is_empty() && !url.is_empty() {
                articles.push(NewsArticle {
                    id: generate_article_id(&url),
                    title,
                    content: String::new(), // 需要单独抓取详情页
                    summary: None,
                    source: NewsSource::CoinDesk,
                    url,
                    published_at,
                    fetched_at: Utc::now(),
                    author: None,
                    tags: Vec::new(),
                    related_symbols: Vec::new(),
                });
            }
        }

        articles
    }

    async fn fetch_article_content(&self, url: &str) -> ETLResult<String> {
        let document = self.fetch_page(url).await?;

        // 选择文章内容（需要根据实际网页结构调整）
        let content_selector = Selector::parse(".article-body, .content-body, article p").unwrap();

        let content = document
            .select(&content_selector)
            .map(|el| el.text().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(extract_text(&content))
    }
}

impl Default for CoinDeskScraper {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NewsScraper for CoinDeskScraper {
    fn source(&self) -> NewsSource {
        NewsSource::CoinDesk
    }

    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        tracing::info!("Fetching latest {} articles from CoinDesk", limit);

        let url = format!("{}/news", self.base_url);
        let document = self.fetch_page(&url).await?;
        let mut articles = self.parse_article_list(&document);

        // 限制数量
        articles.truncate(limit);

        // 抓取文章详情（并发）
        let futures: Vec<_> = articles
            .iter()
            .map(|article| async {
                match self.fetch_article_content(&article.url).await {
                    Ok(content) => Some(content),
                    Err(e) => {
                        tracing::warn!("Failed to fetch article content: {}", e);
                        None
                    }
                }
            })
            .collect();

        let contents = futures::future::join_all(futures).await;

        // 填充内容
        for (article, content) in articles.iter_mut().zip(contents.iter()) {
            if let Some(c) = content {
                article.content = c.clone();
            }
        }

        tracing::info!("Fetched {} articles from CoinDesk", articles.len());
        Ok(articles)
    }

    async fn fetch_by_timerange(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ETLResult<Vec<NewsArticle>> {
        // 抓取最新文章并过滤时间范围
        let articles = self.fetch_latest(100).await?;

        let filtered: Vec<_> = articles
            .into_iter()
            .filter(|article| article.published_at >= start && article.published_at <= end)
            .collect();

        Ok(filtered)
    }

    async fn search(&self, query: &str, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        let url = format!("{}/search?q={}", self.base_url, query);
        let document = self.fetch_page(&url).await?;
        let mut articles = self.parse_article_list(&document);

        articles.truncate(limit);
        Ok(articles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coindesk_scraper() {
        let scraper = CoinDeskScraper::new();
        let result = scraper.fetch_latest(5).await;

        match result {
            Ok(articles) => {
                println!("Fetched {} articles", articles.len());
                for article in articles.iter().take(2) {
                    println!("- {}", article.title);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
