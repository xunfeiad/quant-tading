//! 数据存储和缓存模块

use crate::types::{ETLError, ETLResult, NewsArticle};
use chrono::{DateTime, Utc};
use sqlx::{sqlite::SqlitePool, Row};
use std::path::Path;

/// 数据存储
pub struct Storage {
    pool: SqlitePool,
}

impl Storage {
    /// 创建新的存储实例
    pub async fn new(database_url: &str) -> ETLResult<Self> {
        // 确保数据库文件的目录存在
        if let Some(parent) = Path::new(database_url.trim_start_matches("sqlite:")).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let pool = SqlitePool::connect(database_url)
            .await
            .map_err(|e| ETLError::Database(e.to_string()))?;

        let storage = Self { pool };
        storage.initialize_schema().await?;

        Ok(storage)
    }

    /// 初始化数据库schema
    async fn initialize_schema(&self) -> ETLResult<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                source TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                published_at TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                author TEXT,
                tags TEXT,
                related_symbols TEXT,
                sentiment_score REAL,
                sentiment_confidence REAL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ETLError::Database(e.to_string()))?;

        // 创建索引
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_published_at ON news_articles(published_at);
            CREATE INDEX IF NOT EXISTS idx_source ON news_articles(source);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| ETLError::Database(e.to_string()))?;

        Ok(())
    }

    /// 保存新闻文章
    pub async fn save_article(&self, article: &NewsArticle) -> ETLResult<()> {
        let tags_json = serde_json::to_string(&article.tags)?;
        let symbols_json = serde_json::to_string(&article.related_symbols)?;

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO news_articles
            (id, title, content, summary, source, url, published_at, fetched_at, author, tags, related_symbols)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&article.id)
        .bind(&article.title)
        .bind(&article.content)
        .bind(&article.summary)
        .bind(article.source.to_string())
        .bind(&article.url)
        .bind(article.published_at.to_rfc3339())
        .bind(article.fetched_at.to_rfc3339())
        .bind(&article.author)
        .bind(&tags_json)
        .bind(&symbols_json)
        .execute(&self.pool)
        .await
        .map_err(|e| ETLError::Database(e.to_string()))?;

        Ok(())
    }

    /// 批量保存
    pub async fn save_articles(&self, articles: &[NewsArticle]) -> ETLResult<usize> {
        let mut count = 0;
        for article in articles {
            if self.save_article(article).await.is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// 根据时间范围查询文章
    pub async fn query_by_timerange(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> ETLResult<Vec<NewsArticle>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM news_articles
            WHERE published_at >= ? AND published_at <= ?
            ORDER BY published_at DESC
            "#,
        )
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ETLError::Database(e.to_string()))?;

        let articles: Vec<NewsArticle> = rows
            .iter()
            .filter_map(|row| {
                Some(NewsArticle {
                    id: row.get("id"),
                    title: row.get("title"),
                    content: row.get::<Option<String>, _>("content").unwrap_or_default(),
                    summary: row.get("summary"),
                    source: serde_json::from_str(&row.get::<String, _>("source")).ok()?,
                    url: row.get("url"),
                    published_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("published_at"))
                        .ok()?
                        .with_timezone(&Utc),
                    fetched_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("fetched_at"))
                        .ok()?
                        .with_timezone(&Utc),
                    author: row.get("author"),
                    tags: serde_json::from_str(&row.get::<String, _>("tags")).unwrap_or_default(),
                    related_symbols: serde_json::from_str(&row.get::<String, _>("related_symbols"))
                        .unwrap_or_default(),
                })
            })
            .collect();

        Ok(articles)
    }

    /// 获取最近的文章
    pub async fn query_recent(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        let end = Utc::now();
        let start = end - chrono::Duration::days(7);
        let mut articles = self.query_by_timerange(start, end).await?;
        articles.truncate(limit);
        Ok(articles)
    }

    /// 统计信息
    pub async fn stats(&self) -> ETLResult<StorageStats> {
        let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM news_articles")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| ETLError::Database(e.to_string()))?;

        Ok(StorageStats {
            total_articles: total as usize,
        })
    }
}

#[derive(Debug)]
pub struct StorageStats {
    pub total_articles: usize,
}
