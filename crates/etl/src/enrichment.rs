//! 数据增强和时间对齐模块

use crate::sentiment::SentimentAnalyzer;
use crate::types::{EnrichedMarketData, ETLResult, NewsArticle, SentimentAnalysis};
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// 市场数据（从 exchange crate 可能使用的格式）
#[derive(Debug, Clone)]
pub struct MarketData {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// 数据增强器
pub struct DataEnricher {
    sentiment_analyzer: SentimentAnalyzer,
}

impl DataEnricher {
    pub fn new() -> Self {
        Self {
            sentiment_analyzer: SentimentAnalyzer::new(),
        }
    }

    /// 将新闻数据与市场数据对齐并合并
    pub fn enrich_market_data(
        &self,
        market_data: &[MarketData],
        news_articles: &[NewsArticle],
        time_window_minutes: i64,
    ) -> ETLResult<Vec<EnrichedMarketData>> {
        tracing::info!(
            "Enriching {} market data points with {} news articles",
            market_data.len(),
            news_articles.len()
        );

        // 分析所有新闻的情感
        let sentiments = self.sentiment_analyzer.analyze_batch(news_articles);

        // 创建新闻时间索引
        let news_index = self.build_news_index(news_articles, &sentiments);

        // 为每个市场数据点计算新闻特征
        let enriched: Vec<EnrichedMarketData> = market_data
            .iter()
            .map(|md| {
                let features = self.calculate_news_features(
                    md.timestamp,
                    time_window_minutes,
                    &news_index,
                );

                EnrichedMarketData {
                    timestamp: md.timestamp,
                    open: md.open,
                    high: md.high,
                    low: md.low,
                    close: md.close,
                    volume: md.volume,
                    news_count: features.news_count,
                    avg_sentiment: features.avg_sentiment,
                    positive_ratio: features.positive_ratio,
                    negative_ratio: features.negative_ratio,
                    news_intensity: features.news_intensity,
                    mention_count: features.mention_count,
                    buzz_score: features.buzz_score,
                }
            })
            .collect();

        tracing::info!("Enrichment complete");
        Ok(enriched)
    }

    /// 构建新闻索引（时间 -> 新闻列表）
    fn build_news_index(
        &self,
        articles: &[NewsArticle],
        sentiments: &[SentimentAnalysis],
    ) -> HashMap<DateTime<Utc>, Vec<(NewsArticle, SentimentAnalysis)>> {
        let mut index: HashMap<DateTime<Utc>, Vec<(NewsArticle, SentimentAnalysis)>> =
            HashMap::new();

        for (article, sentiment) in articles.iter().zip(sentiments.iter()) {
            // 向下取整到小时
            let hour = article
                .published_at
                .date_naive()
                .and_hms_opt(article.published_at.hour(), 0, 0)
                .unwrap()
                .and_local_timezone(Utc)
                .unwrap();

            index
                .entry(hour)
                .or_insert_with(Vec::new)
                .push((article.clone(), sentiment.clone()));
        }

        index
    }

    /// 计算指定时间窗口的新闻特征
    fn calculate_news_features(
        &self,
        timestamp: DateTime<Utc>,
        window_minutes: i64,
        news_index: &HashMap<DateTime<Utc>, Vec<(NewsArticle, SentimentAnalysis)>>,
    ) -> NewsFeatures {
        let window_start = timestamp - Duration::minutes(window_minutes);

        // 收集时间窗口内的所有新闻
        let mut relevant_news = Vec::new();

        for hour in 0..=(window_minutes / 60 + 1) {
            let check_time = window_start + Duration::hours(hour);
            let hour_key = check_time
                .date_naive()
                .and_hms_opt(check_time.hour(), 0, 0)
                .unwrap()
                .and_local_timezone(Utc)
                .unwrap();

            if let Some(news_list) = news_index.get(&hour_key) {
                for (article, sentiment) in news_list {
                    if article.published_at >= window_start && article.published_at <= timestamp {
                        relevant_news.push((article, sentiment));
                    }
                }
            }
        }

        if relevant_news.is_empty() {
            return NewsFeatures::default();
        }

        // 计算特征
        let news_count = relevant_news.len();

        let sentiment_scores: Vec<f64> = relevant_news.iter().map(|(_, s)| s.score).collect();

        let avg_sentiment = sentiment_scores.iter().sum::<f64>() / news_count as f64;

        let positive_count = sentiment_scores.iter().filter(|&&s| s > 0.2).count();
        let negative_count = sentiment_scores.iter().filter(|&&s| s < -0.2).count();

        let positive_ratio = positive_count as f64 / news_count as f64;
        let negative_ratio = negative_count as f64 / news_count as f64;

        // 新闻强度 = 新闻量 × 平均情感强度
        let avg_intensity = sentiment_scores.iter().map(|s| s.abs()).sum::<f64>()
            / news_count as f64;
        let news_intensity = news_count as f64 * avg_intensity;

        // 提及次数（包含加密货币实体的新闻数）
        let mention_count = relevant_news
            .iter()
            .filter(|(_, s)| !s.entities.is_empty())
            .count();

        // 热度分数 = (新闻量 / 10) × (1 + |平均情感|) × (1 + 提及率)
        let mention_rate = mention_count as f64 / news_count as f64;
        let buzz_score = (news_count as f64 / 10.0)
            * (1.0 + avg_sentiment.abs())
            * (1.0 + mention_rate);

        NewsFeatures {
            news_count,
            avg_sentiment,
            positive_ratio,
            negative_ratio,
            news_intensity,
            mention_count,
            buzz_score,
        }
    }

    /// 计算滚动新闻特征（用于流式数据）
    pub fn calculate_rolling_features(
        &self,
        market_data: &[MarketData],
        news_articles: &[NewsArticle],
        windows: &[i64], // 多个时间窗口（分钟）
    ) -> ETLResult<Vec<HashMap<String, Vec<f64>>>> {
        let sentiments = self.sentiment_analyzer.analyze_batch(news_articles);
        let news_index = self.build_news_index(news_articles, &sentiments);

        let mut results = Vec::new();

        for md in market_data {
            let mut features = HashMap::new();

            for &window in windows {
                let nf = self.calculate_news_features(md.timestamp, window, &news_index);

                features
                    .entry(format!("news_count_{}", window))
                    .or_insert_with(Vec::new)
                    .push(nf.news_count as f64);

                features
                    .entry(format!("avg_sentiment_{}", window))
                    .or_insert_with(Vec::new)
                    .push(nf.avg_sentiment);

                features
                    .entry(format!("buzz_score_{}", window))
                    .or_insert_with(Vec::new)
                    .push(nf.buzz_score);
            }

            results.push(features);
        }

        Ok(results)
    }
}

impl Default for DataEnricher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default)]
struct NewsFeatures {
    news_count: usize,
    avg_sentiment: f64,
    positive_ratio: f64,
    negative_ratio: f64,
    news_intensity: f64,
    mention_count: usize,
    buzz_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NewsSource;

    #[test]
    fn test_data_enrichment() {
        let enricher = DataEnricher::new();

        // 创建模拟市场数据
        let market_data = vec![
            MarketData {
                timestamp: Utc::now(),
                open: 50000.0,
                high: 51000.0,
                low: 49000.0,
                close: 50500.0,
                volume: 1000000.0,
            },
        ];

        // 创建模拟新闻
        let news = vec![
            NewsArticle {
                id: "1".to_string(),
                title: "Bitcoin surges to new high".to_string(),
                content: "Bullish market sentiment".to_string(),
                summary: None,
                source: NewsSource::CoinDesk,
                url: "http://example.com/1".to_string(),
                published_at: Utc::now() - Duration::minutes(30),
                fetched_at: Utc::now(),
                author: None,
                tags: Vec::new(),
                related_symbols: vec!["BTC".to_string()],
            },
        ];

        let enriched = enricher.enrich_market_data(&market_data, &news, 60).unwrap();

        assert_eq!(enriched.len(), 1);
        println!("News count: {}", enriched[0].news_count);
        println!("Avg sentiment: {}", enriched[0].avg_sentiment);
        println!("Buzz score: {}", enriched[0].buzz_score);
    }
}
