//! 基础量化策略示例
//!
//! 这个示例展示了如何:
//! 1. 准备市场数据
//! 2. 计算技术指标特征
//! 3. 训练随机森林模型
//! 4. 生成交易信号
//! 5. 运行回测

use ml::backtest::{BacktestConfig, BacktestEngine};
use ml::evaluation::Evaluator;
use ml::models::traditional::RandomForestRegressor;
use ml::models::Model;
use ml::preprocessing::{FeatureEngine, Scaler};
use ml::strategy::{PositionManager, SignalGenerator, StrategyConfig};
use ml::timeseries::TimeSeriesSplitter;
use ml::types::{MarketData, Prediction, TradingSignal};
use chrono::{Duration, Utc};
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    println!("=== 量化交易策略示例 ===\n");

    // 1. 生成模拟市场数据
    println!("1. 生成模拟市场数据...");
    let market_data = generate_mock_data(1000);
    println!("   生成了 {} 条数据", market_data.len());

    // 2. 计算技术指标特征
    println!("\n2. 计算技术指标特征...");
    let features = FeatureEngine::compute_technical_indicators(&market_data)?;
    println!("   特征维度: {} x {}", features.nrows(), features.ncols());

    // 3. 准备目标变量（未来收益率）
    println!("\n3. 准备目标变量...");
    let targets = prepare_targets(&market_data, 1);
    println!("   目标变量维度: {} x {}", targets.nrows(), targets.ncols());

    // 4. 数据分割
    println!("\n4. 分割训练集和测试集...");
    let (train_features, val_features, test_features) =
        TimeSeriesSplitter::split_array2(&features, 0.7, 0.15)?;
    let (train_targets, val_targets, test_targets) =
        TimeSeriesSplitter::split_array2(&targets, 0.7, 0.15)?;

    println!("   训练集: {} 样本", train_features.nrows());
    println!("   验证集: {} 样本", val_features.nrows());
    println!("   测试集: {} 样本", test_features.nrows());

    // 5. 数据标准化
    println!("\n5. 数据标准化...");
    let scaler = Scaler::fit(&train_features)?;
    let train_features_norm = scaler.transform(&train_features)?;
    let test_features_norm = scaler.transform(&test_features)?;

    // 6. 训练模型
    println!("\n6. 训练随机森林模型...");
    let mut model = RandomForestRegressor::new(
        50,  // 50 棵树
        10,  // 最大深度 10
        5,   // 最小分割样本数 5
    );

    model.train(&train_features_norm, &train_targets).await?;
    println!("   模型训练完成!");

    // 7. 评估模型
    println!("\n7. 评估模型性能...");
    let predictions = model.predict(&test_features_norm).await?;
    let metrics = Evaluator::evaluate(&test_targets, &predictions)?;

    println!("   MSE: {:.6}", metrics.mse);
    println!("   RMSE: {:.6}", metrics.rmse);
    println!("   MAE: {:.6}", metrics.mae);
    println!("   R²: {:.6}", metrics.r2_score);
    println!("   方向准确率: {:.2}%", metrics.direction_accuracy * 100.0);

    // 8. 生成预测对象
    println!("\n8. 生成交易预测...");
    let test_start_idx = (market_data.len() as f64 * 0.85) as usize;
    let test_market_data = &market_data[test_start_idx..];

    let mut prediction_objects = Vec::new();
    for (i, pred_value) in predictions.column(0).iter().enumerate() {
        prediction_objects.push(Prediction {
            timestamp: test_market_data[i].timestamp,
            price_change: *pred_value,
            signal: TradingSignal::from_prediction(*pred_value),
            confidence: 0.7, // 简化示例，实际应该根据模型计算
        });
    }

    // 9. 回测策略
    println!("\n9. 运行回测...");
    let strategy_config = StrategyConfig {
        buy_threshold: 0.01,     // 1% 收益预期时买入
        sell_threshold: -0.01,   // -1% 收益预期时卖出
        min_confidence: 0.6,
        stop_loss_pct: 0.02,     // 2% 止损
        take_profit_pct: 0.05,   // 5% 止盈
    };

    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        commission_rate: 0.001,  // 0.1% 手续费
        slippage_rate: 0.0005,   // 0.05% 滑点
    };

    let position_manager = PositionManager::new(strategy_config.clone());
    let signal_generator = SignalGenerator::new(strategy_config);

    let mut engine = BacktestEngine::new(
        backtest_config,
        position_manager,
        signal_generator,
    );

    let result = engine.run(test_market_data, &prediction_objects);

    // 10. 显示回测结果
    println!("\n10. 回测结果:");
    result.print_report();

    println!("\n=== 策略示例完成 ===");

    Ok(())
}

/// 生成模拟市场数据
fn generate_mock_data(n: usize) -> Vec<MarketData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    let mut price = 100.0;
    let base_time = Utc::now() - Duration::days(n as i64);

    for i in 0..n {
        // 随机游走 + 趋势
        let trend = (i as f64 / n as f64) * 20.0; // 上升趋势
        let random_change = rng.gen_range(-2.0..2.0);
        price = (price + random_change + 0.1).max(1.0);

        let adjusted_price = price + trend;
        let volatility = rng.gen_range(0.5..2.0);

        data.push(MarketData {
            timestamp: base_time + Duration::hours(i as i64),
            open: adjusted_price,
            high: adjusted_price + volatility,
            low: adjusted_price - volatility,
            close: adjusted_price + rng.gen_range(-0.5..0.5),
            volume: rng.gen_range(1000.0..10000.0),
        });
    }

    data
}

/// 准备目标变量（未来 N 期收益率）
fn prepare_targets(data: &[MarketData], horizon: usize) -> Array2<f64> {
    let n = data.len();
    let mut targets = Array2::<f64>::zeros((n, 1));

    for i in 0..n.saturating_sub(horizon) {
        let current_price = data[i].close;
        let future_price = data[i + horizon].close;
        targets[[i, 0]] = (future_price - current_price) / current_price;
    }

    targets
}
