# ML - 量化交易机器学习引擎

这是一个完整的量化交易机器学习引擎，提供从数据预处理、模型训练、策略生成到回测的全流程支持。

## 功能特性

### 1. 数据预处理
- **特征工程**: 自动计算 20+ 种技术指标
  - 价格相关: 收益率、移动平均线、布林带
  - 动量指标: RSI、MACD
  - 波动率: ATR (平均真实范围)
  - 成交量指标: 成交量移动平均
- **数据标准化**: Z-score 标准化
- **时间序列处理**: 滑动窗口、滞后特征、滚动统计

### 2. 机器学习模型

#### 传统机器学习
- **随机森林** (Random Forest): 适合非线性关系的表格数据
- **线性回归** (Linear Regression): 快速基准模型
- 易于扩展更多模型

#### 深度学习 (基于 PyTorch)
- **LSTM**: 长短期记忆网络，捕捉长期依赖
- **GRU**: 门控循环单元，训练更快
- **支持 GPU 加速**

### 3. 交易策略
- **信号生成器**: 基于模型预测生成交易信号
- **仓位管理**: 自动止损、止盈
- **风险管理**: 仓位大小控制、最大回撤保护
- **策略组合**: 多策略加权投票

### 4. 回测引擎
- 完整的回测框架
- 考虑手续费和滑点
- 丰富的性能指标:
  - 总收益率、年化收益率
  - 夏普比率
  - 最大回撤
  - 胜率、盈亏比

## 快速开始

### 安装依赖

```toml
[dependencies]
ml = { path = "../ml" }
```

### 基本使用

```rust
use ml::*;
use ml::preprocessing::*;
use ml::models::traditional::*;
use ml::models::Model;
use ml::strategy::*;
use ml::backtest::*;

#[tokio::main]
async fn main() -> MLResult<()> {
    // 1. 准备数据
    let market_data = vec![/* 从交易所获取的 OHLCV 数据 */];

    // 2. 特征工程
    let features = FeatureEngine::compute_technical_indicators(&market_data)?;

    // 3. 数据标准化
    let (scaler, normalized_features) = Scaler::fit_transform(&features)?;

    // 4. 准备目标变量（未来收益率）
    let targets = prepare_targets(&market_data, prediction_horizon);

    // 5. 训练模型
    let mut model = RandomForestRegressor::new(100, 10, 5);
    model.train(&normalized_features, &targets).await?;

    // 6. 预测
    let predictions = model.predict(&test_features).await?;

    // 7. 生成交易信号
    let strategy_config = StrategyConfig::default();
    let signal_generator = SignalGenerator::new(strategy_config.clone());

    // 8. 回测
    let backtest_config = BacktestConfig::default();
    let position_manager = PositionManager::new(strategy_config);

    let mut engine = BacktestEngine::new(
        backtest_config,
        position_manager,
        signal_generator,
    );

    let result = engine.run(&market_data, &predictions);
    result.print_report();

    Ok(())
}
```

### 使用深度学习模型

```rust
use ml::models::deep_learning::*;
use ml::models::TimeSeriesModel;
use ml::timeseries::*;

#[tokio::main]
async fn main() -> MLResult<()> {
    // 1. 准备时间序列数据
    let market_data = vec![/* OHLCV 数据 */];
    let features = FeatureEngine::compute_technical_indicators(&market_data)?;

    // 2. 构建序列
    let sequence_length = 60; // 用 60 个时间步预测
    let prediction_horizon = 1; // 预测下一个时间步

    let builder = TimeSeriesBuilder::new(sequence_length, prediction_horizon);
    let (x_train, y_train) = builder.build_sequences(&features, &targets)?;

    // 3. 训练 LSTM 模型
    let config = LSTMConfig {
        input_size: features.ncols() as i64,
        hidden_size: 128,
        num_layers: 2,
        output_size: 1,
        dropout: 0.2,
        learning_rate: 0.001,
        epochs: 100,
        batch_size: 32,
    };

    let mut model = LSTMModel::new(config);
    model.train_sequence(&x_train, &y_train).await?;

    // 4. 保存模型
    model.save("models/lstm_model.pt").await?;

    // 5. 预测
    let predictions = model.predict_sequence(&x_test).await?;

    Ok(())
}
```

## 架构设计

```
ml/
├── src/
│   ├── lib.rs                 # 模块导出和引擎配置
│   ├── types.rs               # 核心类型定义
│   ├── preprocessing.rs       # 数据预处理和特征工程
│   ├── timeseries.rs         # 时间序列数据处理
│   ├── models/
│   │   ├── mod.rs            # 模型接口定义
│   │   ├── traditional.rs    # 传统机器学习模型
│   │   └── deep_learning.rs  # 深度学习模型
│   ├── strategy.rs           # 交易策略和信号生成
│   ├── backtest.rs           # 回测引擎
│   └── evaluation.rs         # 模型评估指标
└── Cargo.toml
```

## 技术指标说明

### 自动计算的技术指标 (20+)

1. **价格变化率**: 日收益率
2. **多周期收益率**: 5日、20日、60日收益率
3. **移动平均线**: MA5、MA20、MA60
4. **成交量均线**: VOL_MA5、VOL_MA20、VOL_MA60
5. **RSI** (相对强弱指数): 14日
6. **MACD** (移动平均收敛散度)
7. **布林带**: 上轨、中轨、下轨
8. **ATR** (平均真实范围): 衡量波动率
9. **价格位置**: 收盘价在当日高低点的位置
10. **成交量变化率**

## 性能优化建议

### 1. GPU 加速
```rust
let config = EngineConfig {
    use_gpu: true,  // 启用 GPU
    ..Default::default()
};
```

### 2. 并行处理
- 使用 `rayon` 进行数据并行处理
- 多个模型可以并行训练

### 3. 特征选择
- 使用相关性分析减少冗余特征
- 使用 PCA 降维

## 最佳实践

### 1. 数据准备
- 至少准备 1-2 年的历史数据
- 确保数据质量（无缺失值、异常值）
- 使用 70% 训练、15% 验证、15% 测试的分割比例

### 2. 模型选择
- 先尝试简单模型（线性回归、随机森林）作为基准
- 数据量充足时再使用深度学习模型
- LSTM 适合捕捉长期趋势，GRU 训练更快

### 3. 策略优化
- 调整买卖阈值以平衡交易频率和准确率
- 设置合理的止损止盈比例
- 使用策略组合提高稳定性

### 4. 风险控制
- 始终设置止损
- 控制单次交易仓位大小
- 监控最大回撤

## 示例项目

完整的使用示例请参考 `examples/` 目录：

```bash
cargo run --example basic_strategy      # 基础策略示例
cargo run --example lstm_prediction     # LSTM 预测示例
cargo run --example ensemble_strategy   # 策略组合示例
```

## 依赖项

主要依赖:
- `ndarray`: 数值计算
- `tch`: PyTorch Rust 绑定 (深度学习)
- `serde`: 序列化/反序列化
- `tokio`: 异步运行时

## 注意事项

1. **深度学习模型**需要安装 PyTorch C++ 库 (libtorch)
2. GPU 支持需要 CUDA 环境
3. 回测结果不代表未来表现，请谨慎使用
4. 实盘交易前务必充分测试

## 路线图

- [ ] 添加更多技术指标
- [ ] 支持 Transformer 模型
- [ ] 实现强化学习策略
- [ ] 添加因子分析工具
- [ ] 支持高频交易策略
- [ ] 集成更多传统机器学习算法

## 许可证

MIT License
