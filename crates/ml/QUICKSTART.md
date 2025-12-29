# ML Crate 快速入门指南

## 5 分钟上手

这个指南将帮助你快速开始使用 ML crate 构建量化交易策略。

## 前置要求

### 基础要求
```bash
# Rust 工具链
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 添加到项目依赖
# 在 Cargo.toml 中:
[dependencies]
ml = { path = "crates/ml" }
```

### 深度学习支持（可选）

如果你想使用 LSTM/GRU 模型，需要安装 PyTorch:

**Linux/macOS**:
```bash
# CPU 版本
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

**Windows**: 访问 https://pytorch.org/ 下载预编译版本

## 第一个策略

### 1. 运行示例

最简单的方式是运行内置示例:

```bash
# 基础策略（使用随机森林）
cargo run --example basic_strategy

# LSTM 深度学习（需要 PyTorch）
cargo run --example lstm_prediction
```

### 2. 创建自己的策略

创建 `src/main.rs`:

```rust
use ml::*;
use ml::preprocessing::*;
use ml::models::traditional::*;
use ml::models::Model;

#[tokio::main]
async fn main() -> MLResult<()> {
    // 1. 准备你的市场数据
    let market_data = vec![
        MarketData {
            timestamp: Utc::now(),
            open: 100.0,
            high: 102.0,
            low: 99.0,
            close: 101.0,
            volume: 10000.0,
        },
        // ... 更多数据
    ];

    // 2. 自动计算技术指标
    let features = FeatureEngine::compute_technical_indicators(&market_data)?;

    // 3. 准备目标（下一期收益率）
    let mut targets = Array2::zeros((features.nrows(), 1));
    for i in 0..targets.nrows() - 1 {
        targets[[i, 0]] = (market_data[i + 1].close - market_data[i].close)
                         / market_data[i].close;
    }

    // 4. 标准化
    let (scaler, features_norm) = Scaler::fit_transform(&features)?;

    // 5. 训练模型
    let mut model = RandomForestRegressor::new(50, 10, 5);
    model.train(&features_norm, &targets).await?;

    // 6. 预测
    let predictions = model.predict(&features_norm).await?;

    println!("预测完成! 前 5 个预测: {:?}",
             predictions.slice(s![0..5, ..]));

    Ok(())
}
```

## 常见使用场景

### 场景 1: 日内交易策略

```rust
use ml::strategy::*;

// 配置激进的日内交易策略
let config = StrategyConfig {
    buy_threshold: 0.005,      // 0.5% 预期收益即买入
    sell_threshold: -0.005,    // -0.5% 预期损失即卖出
    min_confidence: 0.7,       // 要求 70% 置信度
    stop_loss_pct: 0.01,       // 1% 止损
    take_profit_pct: 0.02,     // 2% 止盈
};

let signal_gen = SignalGenerator::new(config);
```

### 场景 2: 长期趋势跟踪

```rust
let config = StrategyConfig {
    buy_threshold: 0.02,       // 2% 预期收益
    sell_threshold: -0.02,
    min_confidence: 0.8,       // 更高的置信度要求
    stop_loss_pct: 0.05,       // 5% 止损
    take_profit_pct: 0.15,     // 15% 止盈
};
```

### 场景 3: 使用 LSTM 预测

```rust
use ml::models::deep_learning::*;
use ml::timeseries::*;

// 准备序列数据
let builder = TimeSeriesBuilder::new(60, 1);  // 60 步预测 1 步
let (x, y) = builder.build_sequences(&features, &targets)?;

// 配置并训练 LSTM
let config = LSTMConfig {
    input_size: 20,
    hidden_size: 128,
    num_layers: 2,
    output_size: 1,
    epochs: 100,
    ..Default::default()
};

let mut model = LSTMModel::new(config);
model.train_sequence(&x, &y).await?;
```

### 场景 4: 策略回测

```rust
use ml::backtest::*;

let backtest_config = BacktestConfig {
    initial_capital: 10000.0,
    commission_rate: 0.001,    // 0.1% 手续费
    slippage_rate: 0.0005,     // 0.05% 滑点
};

let mut engine = BacktestEngine::new(
    backtest_config,
    position_manager,
    signal_generator,
);

let result = engine.run(&market_data, &predictions);
result.print_report();  // 输出详细报告
```

## 工作流程

### 完整的策略开发流程

```
1. 数据收集
   ↓
2. 特征工程（自动完成）
   ↓
3. 模型训练
   ↓
4. 策略回测
   ↓
5. 参数优化
   ↓
6. 实盘交易（谨慎！）
```

### 参数调优建议

#### 模型参数

**随机森林**:
- `n_trees`: 50-200（更多树更稳定但更慢）
- `max_depth`: 5-15（太深容易过拟合）
- `min_samples_split`: 5-20

**LSTM**:
- `hidden_size`: 64-256
- `num_layers`: 2-4
- `dropout`: 0.1-0.3
- `learning_rate`: 0.0001-0.01

#### 策略参数

- **买卖阈值**: 根据资产波动性调整
- **止损**: 1-5% (短线) 或 5-10% (长线)
- **止盈**: 止损的 2-3 倍

## 数据要求

### 最小数据量

- **传统机器学习**: 至少 500-1000 条记录
- **深度学习**: 至少 5000-10000 条记录

### 数据质量

```rust
// 检查数据质量
fn check_data_quality(data: &[MarketData]) {
    for (i, d) in data.iter().enumerate() {
        // 检查缺失值
        assert!(d.close > 0.0, "价格不能为 0");

        // 检查逻辑错误
        assert!(d.high >= d.low, "最高价应 >= 最低价");
        assert!(d.high >= d.close, "最高价应 >= 收盘价");
        assert!(d.low <= d.close, "最低价应 <= 收盘价");
    }
}
```

## 调试技巧

### 1. 启用日志

```rust
// 在 main.rs 开头
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### 2. 检查特征

```rust
let features = FeatureEngine::compute_technical_indicators(&market_data)?;
println!("特征统计:");
println!("  均值: {:?}", features.mean_axis(Axis(0)));
println!("  标准差: {:?}", features.std_axis(Axis(0), 0.0));
```

### 3. 验证预测

```rust
let predictions = model.predict(&test_features).await?;
println!("预测值范围: [{}, {}]",
         predictions.iter().min(),
         predictions.iter().max());
```

## 常见问题

### Q: 模型预测准确率很低怎么办？

A:
1. 检查数据质量
2. 增加训练数据量
3. 尝试不同的特征组合
4. 调整模型参数
5. 考虑使用策略组合

### Q: 回测效果很好但实盘亏损？

A: 这是"过拟合"，建议:
1. 使用更长时间的测试集
2. 交叉验证
3. 增加手续费和滑点
4. 降低交易频率

### Q: LSTM 训练太慢？

A:
1. 减少 `epochs`
2. 增加 `batch_size`
3. 减少 `hidden_size` 或 `num_layers`
4. 使用 GPU (`use_gpu: true`)

### Q: 需要什么样的硬件？

A:
- **最低**: 4GB RAM, 双核 CPU
- **推荐**: 16GB RAM, 四核 CPU
- **深度学习**: + NVIDIA GPU (4GB+ VRAM)

## 下一步

1. 阅读 [完整文档](README.md)
2. 查看 [架构设计](ARCHITECTURE.md)
3. 研究 [示例代码](examples/)
4. 集成你的交易所 API (使用 `exchange` crate)

## 获取帮助

- 查看测试用例了解 API 用法
- 阅读代码注释
- 运行 `cargo doc --open` 查看 API 文档

## 风险提示

这是一个教育/研究工具。**实盘交易有风险，请谨慎使用！**

- 始终从小资金开始
- 充分回测验证
- 设置合理的止损
- 不要投入超过你能承受损失的资金
