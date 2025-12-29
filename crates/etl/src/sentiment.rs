//! 情感分析模块

use crate::types::{Entity, EntityType, ETLResult, NewsArticle, Sentiment, SentimentAnalysis};
use regex::Regex;
use std::collections::HashMap;

/// 情感分析器
pub struct SentimentAnalyzer {
    positive_words: HashMap<String, f64>,
    negative_words: HashMap<String, f64>,
    crypto_entities: HashMap<String, Vec<String>>,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = Self {
            positive_words: HashMap::new(),
            negative_words: HashMap::new(),
            crypto_entities: HashMap::new(),
        };

        analyzer.initialize_dictionaries();
        analyzer
    }

    fn initialize_dictionaries(&mut self) {
        // 正面词汇（加密货币领域）
        let positive_words = vec![
            ("bullish", 2.0),
            ("bull", 2.0),
            ("surge", 2.0),
            ("soar", 2.0),
            ("rally", 2.0),
            ("gain", 1.5),
            ("rise", 1.5),
            ("profit", 1.5),
            ("growth", 1.5),
            ("positive", 1.0),
            ("good", 1.0),
            ("great", 1.5),
            ("excellent", 2.0),
            ("breakthrough", 2.0),
            ("innovation", 1.5),
            ("adoption", 1.5),
            ("institutional", 1.0),
            ("mainstream", 1.0),
            ("record", 1.5),
            ("high", 1.0),
            ("moon", 2.0),
            ("rocket", 2.0),
            ("buy", 1.0),
            ("accumulate", 1.0),
            ("upgrade", 1.5),
            ("success", 1.5),
            ("outperform", 1.5),
        ];

        // 负面词汇
        let negative_words = vec![
            ("bearish", -2.0),
            ("bear", -2.0),
            ("crash", -2.5),
            ("plunge", -2.5),
            ("drop", -1.5),
            ("fall", -1.5),
            ("loss", -1.5),
            ("negative", -1.0),
            ("bad", -1.0),
            ("terrible", -2.0),
            ("scam", -2.5),
            ("fraud", -2.5),
            ("hack", -2.0),
            ("vulnerability", -1.5),
            ("risk", -1.0),
            ("decline", -1.5),
            ("dump", -2.0),
            ("sell", -1.0),
            ("low", -1.0),
            ("concern", -1.0),
            ("warning", -1.5),
            ("crisis", -2.0),
            ("panic", -2.0),
            ("fear", -1.5),
            ("uncertain", -1.0),
            ("volatile", -0.5),
            ("underperform", -1.5),
        ];

        for (word, score) in positive_words {
            self.positive_words.insert(word.to_string(), score);
        }

        for (word, score) in negative_words {
            self.negative_words.insert(word.to_string(), score);
        }

        // 加密货币实体识别
        self.crypto_entities.insert(
            "BTC".to_string(),
            vec!["bitcoin".to_string(), "btc".to_string()],
        );
        self.crypto_entities.insert(
            "ETH".to_string(),
            vec!["ethereum".to_string(), "eth".to_string(), "ether".to_string()],
        );
        self.crypto_entities.insert(
            "USDT".to_string(),
            vec!["tether".to_string(), "usdt".to_string()],
        );
        self.crypto_entities.insert(
            "BNB".to_string(),
            vec!["binance".to_string(), "bnb".to_string()],
        );
        self.crypto_entities.insert(
            "SOL".to_string(),
            vec!["solana".to_string(), "sol".to_string()],
        );
        self.crypto_entities.insert(
            "XRP".to_string(),
            vec!["ripple".to_string(), "xrp".to_string()],
        );
        self.crypto_entities.insert(
            "ADA".to_string(),
            vec!["cardano".to_string(), "ada".to_string()],
        );
    }

    /// 分析新闻文章的情感
    pub fn analyze(&self, article: &NewsArticle) -> ETLResult<SentimentAnalysis> {
        let text = format!("{} {}", article.title, article.content).to_lowercase();
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut positive_score = 0.0;
        let mut negative_score = 0.0;
        let mut keywords = Vec::new();

        // 计算情感分数
        for word in &words {
            let cleaned_word = word.trim_matches(|c: char| !c.is_alphanumeric());

            if let Some(&score) = self.positive_words.get(cleaned_word) {
                positive_score += score;
                keywords.push(cleaned_word.to_string());
            }

            if let Some(&score) = self.negative_words.get(cleaned_word) {
                negative_score += score.abs();
                keywords.push(cleaned_word.to_string());
            }
        }

        // 归一化分数
        let total_words = words.len() as f64;
        let raw_score = (positive_score - negative_score) / total_words.max(1.0);

        // 归一化到 [-1, 1]
        let score = raw_score.max(-1.0).min(1.0);

        // 计算置信度
        let confidence = ((positive_score + negative_score) / total_words.max(1.0))
            .min(1.0);

        // 提取实体
        let entities = self.extract_entities(&text);

        // 去重关键词
        keywords.sort();
        keywords.dedup();
        keywords.truncate(10);

        Ok(SentimentAnalysis {
            score,
            sentiment: Sentiment::from_score(score),
            confidence,
            keywords,
            entities,
        })
    }

    /// 提取命名实体
    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // 提取加密货币
        for (symbol, keywords) in &self.crypto_entities {
            for keyword in keywords {
                if text.contains(keyword) {
                    entities.push(Entity {
                        text: symbol.clone(),
                        entity_type: EntityType::Cryptocurrency,
                    });
                    break;
                }
            }
        }

        // 提取金额
        let number_pattern = Regex::new(r"\$[\d,]+(?:\.\d+)?[KMB]?").unwrap();
        for cap in number_pattern.captures_iter(text) {
            if let Some(matched) = cap.get(0) {
                entities.push(Entity {
                    text: matched.as_str().to_string(),
                    entity_type: EntityType::Number,
                });
            }
        }

        // 去重
        entities.sort_by(|a, b| a.text.cmp(&b.text));
        entities.dedup_by(|a, b| a.text == b.text);

        entities
    }

    /// 批量分析
    pub fn analyze_batch(&self, articles: &[NewsArticle]) -> Vec<SentimentAnalysis> {
        articles
            .iter()
            .filter_map(|article| self.analyze(article).ok())
            .collect()
    }

    /// 计算平均情感
    pub fn calculate_average_sentiment(&self, analyses: &[SentimentAnalysis]) -> f64 {
        if analyses.is_empty() {
            return 0.0;
        }

        let sum: f64 = analyses.iter().map(|a| a.score).sum();
        sum / analyses.len() as f64
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::types::NewsSource;

    #[test]
    fn test_sentiment_analysis() {
        let analyzer = SentimentAnalyzer::new();

        let article = NewsArticle {
            id: "test".to_string(),
            title: "Bitcoin surges to new record high".to_string(),
            content: "Bitcoin rally continues with bullish momentum and positive market sentiment.".to_string(),
            summary: None,
            source: NewsSource::CoinDesk,
            url: "http://example.com".to_string(),
            published_at: Utc::now(),
            fetched_at: Utc::now(),
            author: None,
            tags: Vec::new(),
            related_symbols: Vec::new(),
        };

        let result = analyzer.analyze(&article).unwrap();

        println!("Score: {}", result.score);
        println!("Sentiment: {:?}", result.sentiment);
        println!("Confidence: {}", result.confidence);
        println!("Keywords: {:?}", result.keywords);
        println!("Entities: {:?}", result.entities);

        assert!(result.score > 0.0, "Should be positive");
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new();

        let article = NewsArticle {
            id: "test".to_string(),
            title: "Bitcoin crash: Market panic as prices plunge".to_string(),
            content: "Bearish sentiment dominates as crypto market faces severe decline and investor fear.".to_string(),
            summary: None,
            source: NewsSource::CoinDesk,
            url: "http://example.com".to_string(),
            published_at: Utc::now(),
            fetched_at: Utc::now(),
            author: None,
            tags: Vec::new(),
            related_symbols: Vec::new(),
        };

        let result = analyzer.analyze(&article).unwrap();

        assert!(result.score < 0.0, "Should be negative");
        assert!(matches!(result.sentiment, Sentiment::Negative | Sentiment::VeryNegative));
    }
}
