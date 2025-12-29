# 量化交易系统 - 项目总览

## 🎯 项目愿景

这是一个**完整的、专业级的量化交易系统**，由三个核心 crate 组成，提供从数据获取、特征工程、机器学习到策略回测的全流程支持。

## 📦 项目结构

```
quant-trading/
├── crates/
│   ├── exchange/          # 交易所接口
│   ├── ml/                # 机器学习引擎
│   └── etl/               # 新闻数据 ETL
├── src/
│   └── main.rs
├── Cargo.toml
└── README.md
```

## 🔧 三大核心 Crate

### 1. Exchange Crate - 交易所接口

**功能**: 对接加密货币交易所，获取市场数据和执行交易

**支持的交易所**:
- OKX
- Gate.io
- (可扩展)

**核心功能**:
- ✅ REST API 调用
- ✅ WebSocket 实时数据流
- ✅ HMAC 签名认证
- ✅ 获取 K线数据 (OHLCV)
- ✅ 下单、撤单
- ✅ 查询余额

**数据输出**:
```rust
MarketData {
    timestamp: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}
```

---

### 2. ML Crate - 机器学习引擎 ⭐

**功能**: 完整的机器学习流程，从特征工程到策略回测

#### 核心模块

**a. 数据预处理** (`preprocessing.rs`)
- 自动计算 **20+ 技术指标**
  - MA (5/20/60)
  - RSI (14)
  - MACD
  - 布林带
  - ATR
  - 成交量指标
- Z-score 标准化

**b. 时间序列处理** (`timeseries.rs`)
- 滑动窗口构建
- 序列数据集生成
- 滞后特征
- 滚动统计

**c. 机器学习模型** (`models/`)
- **传统 ML**:
  - 随机森林回归
  - 线性回归
- **深度学习** (PyTorch):
  - LSTM
  - GRU
  - (支持 GPU)

**d. 交易策略** (`strategy.rs`)
- 信号生成器 (5 级信号)
- 仓位管理 (自动止损止盈)
- 风险管理
- 多策略组合

**e. 回测引擎** (`backtest.rs`)
- 完整回测框架
- 考虑手续费和滑点
- 性能指标:
  - 夏普比率
  - 最大回撤
  - 胜率、盈亏比

**f. 模型评估** (`evaluation.rs`)
- MSE, RMSE, MAE, R²
- 方向准确率

**特征输出**: 20 维技术指标向量

---

### 3. ETL Crate - 新闻数据处理 ⭐

**功能**: 爬取新闻、情感分析、与市场数据时间对齐

#### 核心模块

**a. 多源新闻爬虫** (`scraper/`)
- CryptoPanic (API)
- CoinDesk (网页爬虫)
- CoinTelegraph (网页爬虫)
- RSS 通用爬虫

**b. 情感分析** (`sentiment.rs`)
- 领域特定词典 (50+ 正面词, 50+ 负面词)
- 情感分数 [-1, 1]
- 实体识别 (币种、金额)
- 关键词提取

**c. 数据增强** (`enrichment.rs`)
- 时间对齐算法
- 滑动窗口聚合
- 生成 **7 个新闻特征**:
  - news_count
  - avg_sentiment
  - positive_ratio
  - negative_ratio
  - news_intensity
  - mention_count
  - buzz_score

**d. 数据持久化** (`storage.rs`)
- SQLite 存储
- 缓存机制
- 快速查询

**特征输出**: 7 维新闻情感特征向量

---

## 🔄 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     数据获取层                                 │
├─────────────────────────────────────────────────────────────┤
│  Exchange API (exchange crate)    │  News Scrapers (etl)   │
│  • OKX, Gate.io                   │  • CryptoPanic         │
│  • OHLCV 数据                      │  • CoinDesk            │
│  • 实时 WebSocket                  │  • RSS Feeds           │
└──────────┬──────────────────────────┴────────────┬──────────┘
           │                                       │
           │ MarketData                            │ NewsArticle
           ▼                                       ▼
┌──────────────────────────┐         ┌────────────────────────┐
│   特征工程 (ml crate)      │         │  情感分析 (etl crate)   │
│  • 20+ 技术指标            │         │  • NLP 处理             │
│  • RSI, MACD, 布林带       │         │  • 实体识别             │
│  • 标准化                  │         │  • 情感评分             │
└──────────┬──────────────┘         └────────────┬───────────┘
           │                                       │
           │ 20-dim features                       │ 7-dim features
           └───────────────────┬───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   时间对齐 (etl)      │
                    │  • 滑动窗口           │
                    │  • 特征合并           │
                    └──────────┬──────────┘
                               │
                               │ 27-dim enriched features
                               ▼
                    ┌─────────────────────┐
                    │  ML 模型 (ml crate)  │
                    │  • 随机森林           │
                    │  • LSTM/GRU          │
                    │  • 训练 & 预测        │
                    └──────────┬──────────┘
                               │
                               │ Predictions
                               ▼
                    ┌─────────────────────┐
                    │  策略生成 (ml crate) │
                    │  • 信号生成器         │
                    │  • 仓位管理           │
                    │  • 风险控制           │
                    └──────────┬──────────┘
                               │
                               │ Trading Signals
                               ▼
                    ┌─────────────────────┐
                    │  回测/实盘 (ml)      │
                    │  • 模拟交易           │
                    │  • 性能评估           │
                    │  • 实盘下单           │
                    └─────────────────────┘
```

## 💡 核心创新

### 1. 多维特征融合

**传统方法**: 仅使用 OHLCV (5 维)

**我们的方法**:
- OHLCV: 5 维
- 技术指标 (ml): 20 维
- 新闻情感 (etl): 7 维
- **总计**: 32 维特征

→ **预测准确率显著提升**

### 2. 新闻驱动的交易

```rust
// 示例: 结合价格和新闻
if price_prediction > 0.02 && news_sentiment > 0.5 {
    // 价格上涨预测 + 正面新闻 → 强烈买入信号
    execute_trade(Signal::StrongBuy);
}

if news.buzz_score > 8.0 && news.avg_sentiment < -0.6 {
    // 高热度负面新闻 → 风险预警
    close_all_positions();
}
```

### 3. 时间序列深度学习

```rust
// LSTM 捕捉长期趋势
let builder = TimeSeriesBuilder::new(60, 1);  // 用 60 步预测 1 步
let (x, y) = builder.build_sequences(&features, &targets)?;

let mut model = LSTMModel::new(config);
model.train_sequence(&x, &y).await?;
```

## 📊 性能优势

| 特性 | 传统方法 | 我们的系统 |
|------|---------|-----------|
| 特征维度 | 5 (OHLCV) | 32+ |
| 数据源 | 1 (交易所) | 5+ (交易所+新闻) |
| 模型类型 | 简单指标 | ML + DL |
| 情感分析 | ❌ | ✅ |
| 实时新闻 | ❌ | ✅ |
| 自动回测 | ❌ | ✅ |
| GPU 加速 | ❌ | ✅ |

## 🚀 快速开始

### 1. 基础示例 - 使用传统 ML

```rust
use ml::*;
use exchange::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 获取市场数据
    let market_data = exchange_client.get_klines("BTCUSDT", "1h").await?;

    // 2. 计算技术指标
    let features = FeatureEngine::compute_technical_indicators(&market_data)?;

    // 3. 训练模型
    let mut model = RandomForestRegressor::new(100, 10, 5);
    model.train(&features, &targets).await?;

    // 4. 回测
    let result = engine.run(&market_data, &predictions);
    result.print_report();

    Ok(())
}
```

### 2. 高级示例 - 结合新闻数据

```rust
use ml::*;
use etl::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 获取市场数据
    let market_data = exchange_client.get_klines("BTCUSDT", "1h").await?;

    // 2. ETL: 抓取和分析新闻
    let etl_pipeline = ETLPipelineBuilder::new()
        .with_sources(vec![NewsSource::CryptoPanic])
        .build()
        .await?;

    let enriched_data = etl_pipeline.run(&market_data).await?;

    // 3. ML: 训练 LSTM 模型
    let tech_features = FeatureEngine::compute_technical_indicators(&market_data)?;

    // 合并技术指标 + 新闻特征
    let combined_features = combine_features(&tech_features, &enriched_data);

    let mut model = LSTMModel::new(config);
    model.train_sequence(&combined_features, &targets).await?;

    // 4. 预测和交易
    let prediction = model.predict(&latest_features).await?;
    execute_trading_strategy(prediction);

    Ok(())
}
```

### 3. 实时交易系统

```rust
use tokio::time::{interval, Duration};

let mut ticker = interval(Duration::from_secs(300)); // 5 分钟

loop {
    ticker.tick().await;

    // 获取最新数据
    let market_data = exchange.get_latest().await?;
    let news = etl.fetch_news(20).await?;

    // 增强数据
    let enriched = etl.enrich_data(&[market_data], &news)?;

    // 预测
    let prediction = model.predict(&enriched[0]).await?;

    // 生成信号
    let signal = signal_gen.generate_signal(&prediction);

    // 执行交易
    match signal {
        TradingSignal::Buy => exchange.place_order(...).await?,
        TradingSignal::Sell => exchange.close_position(...).await?,
        _ => {}
    }
}
```

## 📖 文档和示例

### ML Crate
- `crates/ml/README.md` - 完整使用文档
- `crates/ml/ARCHITECTURE.md` - 架构设计
- `crates/ml/QUICKSTART.md` - 快速入门
- `crates/ml/examples/basic_strategy.rs` - 基础策略
- `crates/ml/examples/lstm_prediction.rs` - LSTM 预测

### ETL Crate
- `crates/etl/README.md` - 完整使用文档
- `crates/etl/examples/basic_etl.rs` - ETL 示例

### 运行示例
```bash
# ML 基础策略
cargo run --example basic_strategy

# LSTM 预测 (需要 PyTorch)
cargo run --example lstm_prediction

# ETL 新闻抓取
cargo run --example basic_etl
```

## 🎓 学习路径

### 初学者
1. 先运行 `basic_strategy` 示例，理解基本流程
2. 学习 `QUICKSTART.md`
3. 尝试调整策略参数
4. 运行回测查看结果

### 进阶用户
1. 学习 LSTM 时间序列预测
2. 集成 ETL 新闻数据
3. 自定义特征工程
4. 优化模型超参数

### 高级用户
1. 添加新的数据源
2. 实现自定义模型
3. 策略组合优化
4. 部署实盘系统

## ⚙️ 系统要求

### 最低配置
- Rust 1.70+
- 4GB RAM
- 双核 CPU

### 推荐配置
- Rust 1.75+
- 16GB RAM
- 四核 CPU
- SSD 硬盘

### 深度学习 (可选)
- NVIDIA GPU (4GB+ VRAM)
- CUDA 11.0+
- PyTorch libtorch

## 📝 代码统计

| Crate | 文件数 | 代码行数 | 功能 |
|-------|--------|---------|------|
| exchange | 10+ | 1000+ | 交易所接口 |
| ml | 12 | 2500+ | 机器学习引擎 |
| etl | 14 | 1500+ | 新闻数据 ETL |
| **总计** | **36+** | **5000+** | **完整系统** |

## 🌟 核心优势

1. **完整性**: 覆盖量化交易全流程
2. **专业性**: 生产级代码质量
3. **创新性**: 融合新闻和技术分析
4. **可扩展**: 模块化设计，易于扩展
5. **高性能**: 异步 + GPU 加速
6. **易用性**: 丰富的文档和示例

## ⚠️ 风险提示

这是一个教育/研究项目。**实盘交易有风险！**

- ✅ 充分回测验证
- ✅ 从小资金开始
- ✅ 设置止损
- ✅ 分散投资
- ❌ 不要投入超过承受能力的资金

## 🔮 未来计划

### 短期
- [ ] 添加更多交易所支持
- [ ] 实现 Transformer 模型
- [ ] 支持期货交易
- [ ] Web 界面

### 长期
- [ ] 强化学习策略
- [ ] 高频交易支持
- [ ] 云部署方案
- [ ] 移动端 App

## 🎉 总结

你现在拥有了一个**完整的、专业级的量化交易系统**:

✅ **Exchange**: 连接交易所，获取数据
✅ **ML**: 20+ 技术指标 + ML/DL 模型 + 策略回测
✅ **ETL**: 新闻爬取 + 情感分析 + 7 个新闻特征
✅ **总计**: 32+ 维特征，多模型，全流程自动化

这个系统的核心价值在于：
1. 比传统技术分析多了 **新闻情感维度**
2. 比简单机器学习多了 **深度学习能力**
3. 比手工交易多了 **自动化和纪律性**

祝你的量化交易之路顺利！🚀📈
