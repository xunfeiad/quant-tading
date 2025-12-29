# ML Crate 架构设计文档

## 概述

这个 ML crate 是一个完整的量化交易机器学习引擎，提供从数据预处理到策略回测的全流程支持。

## 模块架构

```
ml/
├── src/
│   ├── lib.rs                 # 核心引擎和模块导出
│   ├── types.rs              # 类型定义（错误、信号、预测等）
│   ├── preprocessing.rs      # 数据预处理和特征工程
│   ├── timeseries.rs        # 时间序列数据处理
│   ├── models/              # 机器学习模型
│   │   ├── mod.rs           # 模型 trait 定义
│   │   ├── traditional.rs   # 传统机器学习
│   │   └── deep_learning.rs # 深度学习模型
│   ├── strategy.rs          # 交易策略生成
│   ├── backtest.rs          # 回测引擎
│   └── evaluation.rs        # 评估指标
├── examples/                # 使用示例
└── README.md               # 使用文档
```

## 核心组件

### 1. 数据预处理 (preprocessing.rs)

#### Scaler - 数据标准化器
- **功能**: Z-score 标准化
- **方法**:
  - `fit()`: 从训练数据计算均值和标准差
  - `transform()`: 标准化数据
  - `fit_transform()`: 一步完成拟合和转换
  - `inverse_transform()`: 反标准化

#### FeatureEngine - 特征工程器
- **功能**: 计算技术指标特征
- **指标**:
  - 价格特征: 收益率、移动平均线
  - 动量指标: RSI、MACD
  - 波动率: ATR、布林带
  - 成交量指标: 成交量变化率

### 2. 时间序列处理 (timeseries.rs)

#### TimeSeriesBuilder
- **功能**: 构建时间序列数据集
- **方法**:
  - `build_sequences()`: 创建 (序列, 目标) 数据对
  - `build_sliding_window()`: 创建滑动窗口序列

#### TimeSeriesSplitter
- **功能**: 按时间顺序分割数据
- **方法**:
  - `split()`: 分割向量数据
  - `split_array2()`: 分割 2D 数组
  - `split_array3()`: 分割 3D 数组（时间序列）

#### TimeSeriesAugmentor
- **功能**: 时间序列特征增强
- **方法**:
  - `add_temporal_features()`: 添加时间特征（小时、星期、月份）
  - `add_lag_features()`: 添加滞后特征
  - `add_rolling_features()`: 添加滚动统计特征

### 3. 机器学习模型 (models/)

#### Model Trait (传统模型接口)
```rust
trait Model {
    async fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> MLResult<()>;
    async fn predict(&self, x: &Array2<f64>) -> MLResult<Array2<f64>>;
    async fn save(&self, path: &str) -> MLResult<()>;
    async fn load(path: &str) -> MLResult<Self>;
}
```

#### TimeSeriesModel Trait (时序模型接口)
```rust
trait TimeSeriesModel {
    async fn train_sequence(&mut self, x: &Array3<f64>, y: &Array2<f64>) -> MLResult<()>;
    async fn predict_sequence(&self, x: &Array3<f64>) -> MLResult<Array2<f64>>;
    async fn save(&self, path: &str) -> MLResult<()>;
    async fn load(path: &str) -> MLResult<Self>;
}
```

#### 实现的模型

**传统机器学习**:
- `RandomForestRegressor`: 随机森林回归
  - 参数: 树的数量、最大深度、最小分割样本数
  - 特点: 适合非线性关系，抗过拟合

- `LinearRegression`: 线性回归
  - 使用 SVD 求解最小二乘
  - 特点: 快速基准模型

**深度学习**:
- `LSTMModel`: LSTM 网络
  - 参数: 输入大小、隐藏层大小、层数、dropout
  - 特点: 捕捉长期依赖关系

- `GRUModel`: GRU 网络
  - 类似 LSTM，但参数更少
  - 特点: 训练更快

### 4. 交易策略 (strategy.rs)

#### SignalGenerator - 信号生成器
- **功能**: 根据预测生成交易信号
- **配置**:
  - 买入/卖出阈值
  - 最小置信度
  - 止损/止盈百分比

#### PositionManager - 仓位管理器
- **功能**: 管理交易仓位
- **特性**:
  - 自动止损止盈
  - 仓位状态跟踪
  - 盈亏计算

#### RiskManager - 风险管理器
- **功能**: 风险控制
- **特性**:
  - 仓位大小计算
  - 最大回撤保护

#### StrategyEnsemble - 策略组合
- **功能**: 多策略加权投票
- **方法**: 组合多个策略的信号

### 5. 回测引擎 (backtest.rs)

#### BacktestEngine
- **功能**: 策略回测
- **考虑因素**:
  - 手续费
  - 滑点
  - 资金管理

- **输出指标**:
  - 总收益率、年化收益率
  - 夏普比率
  - 最大回撤
  - 胜率、盈亏比
  - 交易历史
  - 权益曲线

### 6. 评估模块 (evaluation.rs)

#### Evaluator - 评估器
- **回归指标**:
  - MSE (均方误差)
  - RMSE (均方根误差)
  - MAE (平均绝对误差)
  - R² 分数
  - 方向准确率

- **交易指标**:
  - 夏普比率
  - 最大回撤
  - 胜率
  - 盈亏比

## 数据流

```
原始市场数据 (OHLCV)
    ↓
特征工程 (技术指标)
    ↓
数据标准化
    ↓
时间序列构建 (可选)
    ↓
模型训练
    ↓
预测
    ↓
信号生成
    ↓
回测/实盘交易
```

## 使用流程

### 基础流程（传统机器学习）

```rust
// 1. 数据准备
let market_data = fetch_market_data();
let features = FeatureEngine::compute_technical_indicators(&market_data)?;
let targets = prepare_targets(&market_data);

// 2. 数据预处理
let (scaler, features_norm) = Scaler::fit_transform(&features)?;

// 3. 训练模型
let mut model = RandomForestRegressor::new(100, 10, 5);
model.train(&features_norm, &targets).await?;

// 4. 预测
let predictions = model.predict(&test_features).await?;

// 5. 生成信号
let signal_gen = SignalGenerator::new(config);
let signals = signal_gen.generate_signals(&predictions);

// 6. 回测
let mut engine = BacktestEngine::new(config, position_mgr, signal_gen);
let result = engine.run(&market_data, &predictions);
```

### 高级流程（深度学习）

```rust
// 1. 构建时间序列
let builder = TimeSeriesBuilder::new(seq_len, horizon);
let (x, y) = builder.build_sequences(&features, &targets)?;

// 2. 训练 LSTM
let config = LSTMConfig { ... };
let mut model = LSTMModel::new(config);
model.train_sequence(&x_train, &y_train).await?;

// 3. 预测和回测
let predictions = model.predict_sequence(&x_test).await?;
// ... 后续流程相同
```

## 扩展点

### 添加新模型

1. 实现 `Model` 或 `TimeSeriesModel` trait
2. 在 `models/` 目录添加实现
3. 导出到 `models/mod.rs`

### 添加新特征

在 `FeatureEngine::compute_technical_indicators()` 中添加计算逻辑

### 添加新策略

实现自定义 `SignalGenerator` 或继承 `StrategyEnsemble`

## 性能考虑

1. **并行计算**: 使用 `rayon` 进行数据并行
2. **GPU 加速**: 深度学习模型支持 CUDA
3. **内存优化**: 使用 `ndarray` 的视图避免复制
4. **异步 I/O**: 所有模型操作都是异步的

## 依赖关系

- 数值计算: `ndarray`, `ndarray-linalg`
- 深度学习: `tch` (PyTorch bindings)
- 异步: `tokio`, `async-trait`
- 序列化: `serde`, `bincode`
- 统计: `statrs`

## 测试策略

每个模块都包含单元测试:
```bash
cargo test -p ml
```

运行示例:
```bash
cargo run --example basic_strategy
cargo run --example lstm_prediction
```

## 未来改进

1. 添加更多模型（XGBoost, LightGBM）
2. 实现 Transformer 用于序列预测
3. 添加强化学习支持
4. 集成因子分析
5. 支持高频交易策略
6. 添加模型解释性工具
