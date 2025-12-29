//! 基础 ETL 示例
//!
//! 展示如何使用 ETL crate 抓取新闻并增强交易数据

use chrono::{Duration, Utc};
use etl::{
    enrichment::MarketData,
    pipeline::ETLPipelineBuilder,
    types::NewsSource,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== ETL 基础示例 ===\n");

    // 1. 创建 ETL 管道
    println!("1. 创建 ETL 管道...");
    let pipeline = ETLPipelineBuilder::new()
        .with_sources(vec![
            NewsSource::CryptoPanic,
            NewsSource::CoinDesk,
        ])
        .with_time_window(60) // 1小时窗口
        .with_database("sqlite:data/etl.db".to_string())
        .enable_cache(true)
        .build()
        .await?;

    println!("   ✓ 管道创建完成\n");

    // 2. 抓取新闻
    println!("2. 抓取加密货币新闻...");
    let news = pipeline.fetch_news(20).await?;

    println!("   抓取到 {} 条新闻", news.len());

    // 显示前 5 条新闻
    println!("\n   最新新闻:");
    for (i, article) in news.iter().take(5).enumerate() {
        println!("   {}. [{}] {}",
            i + 1,
            article.source,
            article.title
        );
        println!("      时间: {}", article.published_at.format("%Y-%m-%d %H:%M"));
        println!("      URL: {}\n", article.url);
    }

    // 3. 情感分析
    println!("3. 分析新闻情感...");
    let sentiments = pipeline.analyze_sentiment(&news);

    let avg_sentiment: f64 = sentiments.iter().map(|s| s.score).sum::<f64>()
        / sentiments.len().max(1) as f64;

    println!("   平均情感分数: {:.3}", avg_sentiment);
    println!("   情感分布:");

    let very_positive = sentiments.iter().filter(|s| s.score > 0.5).count();
    let positive = sentiments.iter().filter(|s| s.score > 0.2 && s.score <= 0.5).count();
    let neutral = sentiments.iter().filter(|s| s.score >= -0.2 && s.score <= 0.2).count();
    let negative = sentiments.iter().filter(|s| s.score >= -0.5 && s.score < -0.2).count();
    let very_negative = sentiments.iter().filter(|s| s.score < -0.5).count();

    println!("     非常正面: {} ({:.1}%)", very_positive, very_positive as f64 / sentiments.len() as f64 * 100.0);
    println!("     正面: {} ({:.1}%)", positive, positive as f64 / sentiments.len() as f64 * 100.0);
    println!("     中性: {} ({:.1}%)", neutral, neutral as f64 / sentiments.len() as f64 * 100.0);
    println!("     负面: {} ({:.1}%)", negative, negative as f64 / sentiments.len() as f64 * 100.0);
    println!("     非常负面: {} ({:.1}%)\n", very_negative, very_negative as f64 / sentiments.len() as f64 * 100.0);

    // 4. 创建模拟市场数据
    println!("4. 准备市场数据...");
    let market_data = generate_mock_market_data(24); // 24 小时数据
    println!("   生成了 {} 个市场数据点\n", market_data.len());

    // 5. 数据增强
    println!("5. 使用新闻特征增强市场数据...");
    let enriched = pipeline.enrich_data(&market_data, &news)?;

    println!("   增强完成! 添加了以下特征:");
    println!("     - news_count: 新闻数量");
    println!("     - avg_sentiment: 平均情感分数");
    println!("     - positive_ratio: 正面新闻比例");
    println!("     - negative_ratio: 负面新闻比例");
    println!("     - news_intensity: 新闻强度");
    println!("     - mention_count: 提及次数");
    println!("     - buzz_score: 热度分数\n");

    // 显示前 5 个增强数据点
    println!("   示例数据 (前 5 个时间点):");
    for (i, data) in enriched.iter().take(5).enumerate() {
        println!("   {}. 时间: {}", i + 1, data.timestamp.format("%H:%M"));
        println!("      价格: ${:.2}", data.close);
        println!("      新闻数: {}", data.news_count);
        println!("      情感: {:.3}", data.avg_sentiment);
        println!("      热度: {:.3}\n", data.buzz_score);
    }

    // 6. 显示统计信息
    if let Some(stats) = pipeline.storage_stats().await? {
        println!("6. 数据库统计:");
        println!("   {}\n", stats);
    }

    println!("=== ETL 示例完成 ===");
    println!("\n提示:");
    println!("  - 新闻数据已保存到 data/etl.db");
    println!("  - 可以使用 enriched 数据训练 ML 模型");
    println!("  - 运行多次会积累更多历史数据");

    Ok(())
}

fn generate_mock_market_data(hours: usize) -> Vec<MarketData> {
    let mut data = Vec::new();
    let now = Utc::now();
    let mut price = 50000.0;

    for i in 0..hours {
        // 模拟价格波动
        let change = (i as f64 * 0.1).sin() * 500.0;
        price += change;

        data.push(MarketData {
            timestamp: now - Duration::hours((hours - i) as i64),
            open: price - 100.0,
            high: price + 200.0,
            low: price - 200.0,
            close: price,
            volume: 1000000.0 + (i as f64 * 10000.0),
        });
    }

    data
}
