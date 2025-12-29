//! LSTM 深度学习预测示例
//!
//! 这个示例展示了如何使用 LSTM 模型进行时间序列预测

use ml::models::deep_learning::{LSTMConfig, LSTMModel};
use ml::models::TimeSeriesModel;
use ml::preprocessing::{FeatureEngine, Scaler};
use ml::timeseries::{TimeSeriesBuilder, TimeSeriesSplitter};
use ml::types::MarketData;
use chrono::{Duration, Utc};
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== LSTM 时间序列预测示例 ===\n");

    // 1. 生成模拟数据
    println!("1. 生成模拟市场数据...");
    let market_data = generate_mock_data(500);
    println!("   数据量: {}", market_data.len());

    // 2. 计算技术指标
    println!("\n2. 计算技术指标特征...");
    let features = FeatureEngine::compute_technical_indicators(&market_data)?;
    println!("   特征数: {}", features.ncols());

    // 3. 准备目标变量
    println!("\n3. 准备目标变量...");
    let targets = prepare_targets(&market_data, 1);

    // 4. 标准化
    println!("\n4. 数据标准化...");
    let (scaler, features_norm) = Scaler::fit_transform(&features)?;
    let (target_scaler, targets_norm) = Scaler::fit_transform(&targets)?;

    // 5. 构建时间序列
    println!("\n5. 构建时间序列数据...");
    let sequence_length = 60; // 使用 60 个时间步
    let prediction_horizon = 1;

    let builder = TimeSeriesBuilder::new(sequence_length, prediction_horizon);
    let (x, y) = builder.build_sequences(&features_norm, &targets_norm)?;

    println!("   序列形状: {:?}", x.shape());
    println!("   目标形状: {:?}", y.shape());

    // 6. 分割数据集
    println!("\n6. 分割训练集和测试集...");
    let (x_train, x_val, x_test) = TimeSeriesSplitter::split_array3(&x, 0.7, 0.15)?;
    let (y_train, y_val, y_test) = TimeSeriesSplitter::split_array2(&y, 0.7, 0.15)?;

    println!("   训练集: {} 样本", x_train.shape()[0]);
    println!("   验证集: {} 样本", x_val.shape()[0]);
    println!("   测试集: {} 样本", x_test.shape()[0]);

    // 7. 配置 LSTM 模型
    println!("\n7. 配置 LSTM 模型...");
    let config = LSTMConfig {
        input_size: features.ncols() as i64,
        hidden_size: 64,
        num_layers: 2,
        output_size: 1,
        dropout: 0.2,
        learning_rate: 0.001,
        epochs: 50,  // 为了演示，减少训练轮数
        batch_size: 32,
    };

    println!("   配置: {:?}", config);

    // 8. 训练模型
    println!("\n8. 训练 LSTM 模型...");
    println!("   (这可能需要几分钟...)");

    let mut model = LSTMModel::new(config);

    match model.train_sequence(&x_train, &y_train).await {
        Ok(_) => println!("   ✓ 模型训练完成!"),
        Err(e) => {
            println!("   ✗ 训练失败: {}", e);
            println!("\n注意: LSTM 需要安装 PyTorch (libtorch)");
            println!("请访问: https://pytorch.org/get-started/locally/");
            println!("\n如果您只想测试基本功能，请运行:");
            println!("  cargo run --example basic_strategy");
            return Ok(());
        }
    }

    // 9. 预测
    println!("\n9. 进行预测...");
    let predictions = model.predict_sequence(&x_test).await?;

    // 10. 反标准化预测结果
    let predictions_denorm = target_scaler.inverse_transform(&predictions)?;
    let y_test_denorm = target_scaler.inverse_transform(&y_test)?;

    // 11. 评估
    println!("\n10. 评估模型...");
    use ml::evaluation::Evaluator;
    let metrics = Evaluator::evaluate(&y_test_denorm, &predictions_denorm)?;

    println!("   RMSE: {:.6}", metrics.rmse);
    println!("   MAE: {:.6}", metrics.mae);
    println!("   R²: {:.6}", metrics.r2_score);
    println!("   方向准确率: {:.2}%", metrics.direction_accuracy * 100.0);

    // 12. 保存模型
    println!("\n11. 保存模型...");
    model.save("models/lstm_model.pt").await?;
    println!("   ✓ 模型已保存到 models/lstm_model.pt");

    println!("\n=== LSTM 预测示例完成 ===");

    Ok(())
}

fn generate_mock_data(n: usize) -> Vec<MarketData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    let mut price = 100.0;
    let base_time = Utc::now() - Duration::days(n as i64);

    for i in 0..n {
        let trend = (i as f64 / n as f64) * 30.0;
        let seasonal = 5.0 * ((i as f64 * 0.1).sin());
        let noise = rng.gen_range(-1.0..1.0);

        price = price + 0.1 * (trend + seasonal + noise - price) / 10.0;
        let volatility = rng.gen_range(0.5..1.5);

        data.push(MarketData {
            timestamp: base_time + Duration::hours(i as i64),
            open: price,
            high: price + volatility,
            low: price - volatility,
            close: price + rng.gen_range(-0.3..0.3),
            volume: rng.gen_range(5000.0..15000.0),
        });
    }

    data
}

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
