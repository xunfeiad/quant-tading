# ETL - 加密货币新闻数据 ETL 管道

这是一个专门为量化交易设计的 ETL (Extract, Transform, Load) crate，用于从多个新闻源抓取数据、进行情感分析，并与交易所的市场数据时间对齐，为机器学习模型提供额外的特征。

## 为什么需要新闻数据？

交易所提供的数据（OHLCV）只包含价格和成交量，缺少影响市场的重要因素：

- **新闻事件**: 政策变化、黑客事件、机构采用等
- **市场情绪**: 投资者的乐观/悲观情绪
- **舆论热度**: 某个币种的讨论热度
- **关键实体**: 重要人物、机构的动向

通过添加这些特征，ML 模型可以做出更准确的预测。

## 核心功能

### 1. 多源新闻抓取 ✓

支持多个主流加密货币新闻源：

- **CryptoPanic** - 聚合新闻 API
- **CoinDesk** - 权威新闻网站
- **CoinTelegraph** - 行业新闻
- **RSS 源** - 支持任意 RSS feed

#### 特性：
- 并发抓取多个数据源
- 自动去重（基于 URL）
- 限流保护（避免被封 IP）
- 自动重试机制

### 2. NLP 情感分析 ✓

基于词典的情感分析系统：

- **领域专用词典**: 针对加密货币领域优化
- **情感分数**: [-1.0, 1.0] 范围
- **情感分类**: VeryPositive, Positive, Neutral, Negative, VeryNegative
- **关键词提取**: 识别影响情感的关键词
- **实体识别**: 自动识别币种、金额等

#### 示例：
```
标题: "Bitcoin surges to new record high"
情感分数: 0.75 (Positive)
关键词: ["surge", "record", "high", "bitcoin"]
实体: [BTC]
```

### 3. 时间对齐和特征生成 ✓

将新闻数据与市场数据按时间对齐：

#### 生成的特征：
- `news_count`: 时间窗口内的新闻数量
- `avg_sentiment`: 平均情感分数
- `positive_ratio`: 正面新闻占比
- `negative_ratio`: 负面新闻占比
- `news_intensity`: 新闻强度（数量 × 情感强度）
- `mention_count`: 币种被提及次数
- `buzz_score`: 综合热度分数

#### 时间窗口：
- 可配置（1小时、4小时、1天等）
- 滑动窗口计算
- 支持多窗口特征

### 4. 数据持久化 ✓

- **SQLite 数据库**: 存储所有抓取的新闻
- **缓存机制**: 避免重复抓取
- **时间索引**: 快速查询历史数据

## 快速开始

### 基础用法

```rust
use etl::{
    enrichment::MarketData,
    pipeline::ETLPipelineBuilder,
    types::NewsSource,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 创建 ETL 管道
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

    // 2. 抓取新闻
    let news = pipeline.fetch_news(50).await?;
    println!("抓取到 {} 条新闻", news.len());

    // 3. 情感分析
    let sentiments = pipeline.analyze_sentiment(&news);

    // 4. 增强市场数据
    let market_data: Vec<MarketData> = /* 从交易所获取 */;
    let enriched = pipeline.enrich_data(&market_data, &news)?;

    // 5. 使用增强数据训练 ML 模型
    // enriched 包含原始 OHLCV + 新闻特征

    Ok(())
}
```

### 与 ML Crate 集成

```rust
use etl::pipeline::ETLPipelineBuilder;
use ml::preprocessing::FeatureEngine;

// 1. 获取增强的市场数据
let pipeline = ETLPipelineBuilder::new().build().await?;
let enriched_data = pipeline.run(&market_data).await?;

// 2. 提取特征
let mut features = Vec::new();
for data in &enriched_data {
    // 技术指标特征
    let tech_features = FeatureEngine::compute_technical_indicators(&market_data)?;

    // 新闻特征
    let news_features = vec![
        data.news_count as f64,
        data.avg_sentiment,
        data.positive_ratio,
        data.negative_ratio,
        data.news_intensity,
        data.buzz_score,
    ];

    // 合并特征
    features.push([tech_features, news_features].concat());
}

// 3. 训练模型
let mut model = RandomForestRegressor::new(100, 10, 5);
model.train(&features, &targets).await?;
```

### 实时数据流

```rust
use tokio::time::{interval, Duration};

// 每 5 分钟抓取一次新闻
let mut ticker = interval(Duration::from_secs(300));

loop {
    ticker.tick().await;

    // 抓取最新新闻
    let news = pipeline.fetch_news(20).await?;

    // 获取最新市场数据
    let latest_market_data = exchange.get_latest_kline().await?;

    // 增强数据
    let enriched = pipeline.enrich_data(&[latest_market_data], &news)?;

    // 使用增强数据做预测
    let prediction = model.predict(&enriched[0]).await?;

    // 根据预测执行交易
    // ...
}
```

## 数据源配置

### CryptoPanic (推荐)

免费 API，需要注册获取 API key：

```bash
export CRYPTOPANIC_API_KEY="your_api_key"
```

访问: https://cryptopanic.com/developers/api/

### 其他数据源

大多数其他数据源使用网页抓取，无需 API key：

```rust
let pipeline = ETLPipelineBuilder::new()
    .with_sources(vec![
        NewsSource::CoinDesk,
        NewsSource::CoinTelegraph,
        NewsSource::Bitcoin_com,
        NewsSource::TheBlock,
    ])
    .build()
    .await?;
```

## 架构设计

```
ETL Pipeline
    │
    ├─ Extract (抓取)
    │   ├─ CryptoPanic API
    │   ├─ CoinDesk Scraper
    │   ├─ CoinTelegraph Scraper
    │   └─ RSS Feeds
    │
    ├─ Transform (转换)
    │   ├─ HTML 清理
    │   ├─ 情感分析
    │   ├─ 实体识别
    │   └─ 时间对齐
    │
    └─ Load (加载)
        ├─ SQLite 存储
        ├─ 特征计算
        └─ 数据增强
```

## 性能优化

### 1. 并发抓取
```rust
// 自动并发抓取多个数据源
let pipeline = ETLPipelineBuilder::new()
    .with_sources(vec![/* 多个源 */])
    .build()
    .await?;

// 内部使用 tokio 并发执行
```

### 2. 缓存机制
```rust
// 启用数据库缓存
let pipeline = ETLPipelineBuilder::new()
    .enable_cache(true)
    .build()
    .await?;

// 从缓存加载而不是重新抓取
let enriched = pipeline
    .enrich_from_cache(&market_data, 24) // 使用过去 24 小时的缓存
    .await?;
```

### 3. 限流保护
```rust
// 自动限流，避免被封 IP
// 默认: 每分钟最多 60 个请求
```

## 情感分析详解

### 词典内容

**正面词汇** (部分):
- "bullish", "surge", "rally", "gain" → 强正面
- "positive", "good", "growth" → 中等正面
- "adoption", "institutional" → 轻度正面

**负面词汇** (部分):
- "crash", "scam", "fraud" → 强负面
- "drop", "decline", "hack" → 中等负面
- "risk", "concern", "volatile" → 轻度负面

### 情感计算

```
情感分数 = (正面词权重和 - 负面词权重和) / 总词数
归一化到 [-1, 1]
```

### 置信度

```
置信度 = (正面词权重 + 负面词权重) / 总词数
表示情感判断的可靠性
```

## 运行示例

```bash
# 基础示例
cargo run --example basic_etl

# 输出示例:
# Fetched 45 articles
# Average sentiment: 0.23 (Positive)
# Enriched 24 market data points
# Added 7 news features per data point
```

## 数据库结构

```sql
CREATE TABLE news_articles (
    id TEXT PRIMARY KEY,              -- SHA256(URL)
    title TEXT NOT NULL,
    content TEXT,
    source TEXT NOT NULL,             -- 新闻源
    url TEXT UNIQUE,
    published_at TIMESTAMP,           -- 发布时间
    fetched_at TIMESTAMP,             -- 抓取时间
    sentiment_score REAL,             -- 情感分数
    sentiment_confidence REAL         -- 置信度
);

CREATE INDEX idx_published_at ON news_articles(published_at);
```

## 扩展指南

### 添加新的数据源

1. 实现 `NewsScraper` trait:

```rust
use async_trait::async_trait;

pub struct MyNewsScraper {
    client: reqwest::Client,
}

#[async_trait]
impl NewsScraper for MyNewsScraper {
    fn source(&self) -> NewsSource {
        NewsSource::Custom
    }

    async fn fetch_latest(&self, limit: usize) -> ETLResult<Vec<NewsArticle>> {
        // 实现抓取逻辑
    }

    // 实现其他方法...
}
```

2. 注册到工厂:

```rust
// 在 scraper/mod.rs 的 ScraperFactory::create() 中添加
NewsSource::MySource => Box::new(MyNewsScraper::new()),
```

### 自定义情感分析

```rust
// 添加新词汇到词典
let mut analyzer = SentimentAnalyzer::new();
analyzer.add_positive_word("hodl", 1.5);
analyzer.add_negative_word("dump", -2.0);

// 使用自定义分析器
let sentiment = analyzer.analyze(&article)?;
```

## 最佳实践

### 1. 数据质量
- 定期清理旧数据（保留 30-90 天）
- 检查抓取成功率
- 验证情感分析准确性

### 2. 性能
- 使用缓存减少 API 调用
- 合理设置时间窗口（不要太小）
- 限制并发请求数

### 3. 可靠性
- 处理网络错误和超时
- 数据源失败时使用备用源
- 记录日志便于调试

## 注意事项

1. **爬虫礼仪**: 遵守网站的 robots.txt
2. **API 限制**: 注意各数据源的速率限制
3. **数据延迟**: 新闻可能有几分钟到几小时的延迟
4. **情感准确性**: 基于词典的方法有局限性，考虑使用深度学习模型

## 未来改进

- [ ] 支持更多新闻源（Twitter, Reddit）
- [ ] 深度学习情感分析（BERT）
- [ ] 多语言支持
- [ ] 事件检测和分类
- [ ] 影响力评分（新闻源权重）
- [ ] 实时流处理（Kafka/Redis Streams）

## 许可证

MIT License
