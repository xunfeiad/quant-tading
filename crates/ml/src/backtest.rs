//! 回测引擎

use crate::evaluation::Evaluator;
use crate::strategy::{PositionManager, SignalGenerator, TradeAction};
use crate::types::{MarketData, Position, Prediction, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 回测配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// 初始资金
    pub initial_capital: f64,
    /// 手续费率（每次交易）
    pub commission_rate: f64,
    /// 滑点（价格滑动百分比）
    pub slippage_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission_rate: 0.001,  // 0.1%
            slippage_rate: 0.0005,   // 0.05%
        }
    }
}

/// 交易记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub action: TradeAction,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub pnl: Option<f64>,
}

/// 回测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// 最终资金
    pub final_capital: f64,
    /// 总收益率
    pub total_return: f64,
    /// 年化收益率
    pub annualized_return: f64,
    /// 夏普比率
    pub sharpe_ratio: f64,
    /// 最大回撤
    pub max_drawdown: f64,
    /// 总交易次数
    pub total_trades: usize,
    /// 胜率
    pub win_rate: f64,
    /// 盈亏比
    pub profit_factor: f64,
    /// 交易历史
    pub trades: Vec<Trade>,
    /// 权益曲线
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
}

impl BacktestResult {
    /// 打印回测报告
    pub fn print_report(&self) {
        println!("========== 回测报告 ==========");
        println!("最终资金: ${:.2}", self.final_capital);
        println!("总收益率: {:.2}%", self.total_return * 100.0);
        println!("年化收益率: {:.2}%", self.annualized_return * 100.0);
        println!("夏普比率: {:.4}", self.sharpe_ratio);
        println!("最大回撤: {:.2}%", self.max_drawdown * 100.0);
        println!("总交易次数: {}", self.total_trades);
        println!("胜率: {:.2}%", self.win_rate * 100.0);
        println!("盈亏比: {:.4}", self.profit_factor);
        println!("==============================");
    }
}

/// 回测引擎
pub struct BacktestEngine {
    config: BacktestConfig,
    position_manager: PositionManager,
    signal_generator: SignalGenerator,
    trades: Vec<Trade>,
    equity_curve: Vec<(DateTime<Utc>, f64)>,
    current_capital: f64,
}

impl BacktestEngine {
    pub fn new(
        config: BacktestConfig,
        position_manager: PositionManager,
        signal_generator: SignalGenerator,
    ) -> Self {
        let initial_capital = config.initial_capital;
        Self {
            config,
            position_manager,
            signal_generator,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            current_capital: initial_capital,
        }
    }

    /// 运行回测
    pub fn run(
        &mut self,
        market_data: &[MarketData],
        predictions: &[Prediction],
    ) -> BacktestResult {
        assert_eq!(
            market_data.len(),
            predictions.len(),
            "市场数据和预测数据长度必须一致"
        );

        self.equity_curve.clear();
        self.trades.clear();
        self.current_capital = self.config.initial_capital;

        for (data, pred) in market_data.iter().zip(predictions.iter()) {
            self.process_timestep(data, pred);
        }

        // 如果最后还有持仓，平仓
        if self.position_manager.current_position() != Position::Neutral {
            let last_price = market_data.last().unwrap().close;
            self.close_final_position(market_data.last().unwrap().timestamp, last_price);
        }

        self.calculate_results(market_data)
    }

    fn process_timestep(&mut self, data: &MarketData, prediction: &Prediction) {
        let signal = self.signal_generator.generate_signal(prediction);
        let current_price = self.apply_slippage(data.close, &signal);

        // 更新持仓
        if let Some(action) = self.position_manager.update_position(signal, current_price) {
            self.execute_trade(data.timestamp, action, current_price);
        }

        // 记录权益
        let equity = self.calculate_equity(current_price);
        self.equity_curve.push((data.timestamp, equity));
    }

    fn execute_trade(&mut self, timestamp: DateTime<Utc>, action: TradeAction, price: f64) {
        let commission = self.current_capital * self.config.commission_rate;

        match action {
            TradeAction::OpenLong { price } => {
                let quantity = (self.current_capital - commission) / price;
                self.current_capital = 0.0; // 全仓

                self.trades.push(Trade {
                    timestamp,
                    action,
                    price,
                    quantity,
                    commission,
                    pnl: None,
                });
            }
            TradeAction::OpenShort { price } => {
                let quantity = (self.current_capital - commission) / price;
                self.current_capital = 0.0; // 全仓

                self.trades.push(Trade {
                    timestamp,
                    action,
                    price,
                    quantity,
                    commission,
                    pnl: None,
                });
            }
            TradeAction::Close { position, price } => {
                if let Some(entry_trade) = self.trades.last() {
                    let entry_price = entry_trade.price;
                    let quantity = entry_trade.quantity;

                    let pnl = match position {
                        Position::Long => (price - entry_price) * quantity,
                        Position::Short => (entry_price - price) * quantity,
                        Position::Neutral => 0.0,
                    };

                    self.current_capital = price * quantity - commission + pnl;

                    self.trades.push(Trade {
                        timestamp,
                        action,
                        price,
                        quantity,
                        commission,
                        pnl: Some(pnl),
                    });
                }
            }
        }
    }

    fn close_final_position(&mut self, timestamp: DateTime<Utc>, price: f64) {
        let position = self.position_manager.current_position();
        let action = TradeAction::Close { position, price };
        self.execute_trade(timestamp, action, price);
        self.position_manager.close_position();
    }

    fn calculate_equity(&self, current_price: f64) -> f64 {
        match self.position_manager.current_position() {
            Position::Neutral => self.current_capital,
            Position::Long | Position::Short => {
                if let Some(entry_trade) = self.trades.last() {
                    let entry_price = entry_trade.price;
                    let quantity = entry_trade.quantity;

                    let unrealized_pnl = match self.position_manager.current_position() {
                        Position::Long => (current_price - entry_price) * quantity,
                        Position::Short => (entry_price - current_price) * quantity,
                        Position::Neutral => 0.0,
                    };

                    current_price * quantity + unrealized_pnl
                } else {
                    self.current_capital
                }
            }
        }
    }

    fn apply_slippage(&self, price: f64, signal: &TradingSignal) -> f64 {
        match signal {
            TradingSignal::StrongBuy | TradingSignal::Buy => {
                price * (1.0 + self.config.slippage_rate)
            }
            TradingSignal::StrongSell | TradingSignal::Sell => {
                price * (1.0 - self.config.slippage_rate)
            }
            TradingSignal::Hold => price,
        }
    }

    fn calculate_results(&self, market_data: &[MarketData]) -> BacktestResult {
        let final_capital = self.equity_curve.last().map(|(_, e)| *e).unwrap_or(0.0);
        let total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital;

        // 计算年化收益率
        let days = if market_data.len() > 1 {
            let start = market_data.first().unwrap().timestamp;
            let end = market_data.last().unwrap().timestamp;
            (end - start).num_days() as f64
        } else {
            1.0
        };
        let years = days / 365.0;
        let annualized_return = if years > 0.0 {
            ((1.0 + total_return).powf(1.0 / years)) - 1.0
        } else {
            0.0
        };

        // 计算每日收益率
        let mut daily_returns = Vec::new();
        for i in 1..self.equity_curve.len() {
            let prev_equity = self.equity_curve[i - 1].1;
            let curr_equity = self.equity_curve[i].1;
            if prev_equity > 0.0 {
                daily_returns.push((curr_equity - prev_equity) / prev_equity);
            }
        }

        let sharpe_ratio = Evaluator::sharpe_ratio(&daily_returns, 0.0);

        // 计算最大回撤
        let equity_values: Vec<f64> = self.equity_curve.iter().map(|(_, e)| *e).collect();
        let max_drawdown = Evaluator::max_drawdown(&equity_values);

        // 计算交易统计
        let closed_trades: Vec<&Trade> = self
            .trades
            .iter()
            .filter(|t| t.pnl.is_some())
            .collect();

        let trade_returns: Vec<f64> = closed_trades
            .iter()
            .filter_map(|t| {
                t.pnl.map(|pnl| {
                    pnl / (t.price * t.quantity)
                })
            })
            .collect();

        let win_rate = Evaluator::win_rate(&trade_returns);
        let profit_factor = Evaluator::profit_factor(&trade_returns);

        BacktestResult {
            final_capital,
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            total_trades: closed_trades.len(),
            win_rate,
            profit_factor,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::StrategyConfig;

    #[test]
    fn test_backtest_engine() {
        let backtest_config = BacktestConfig::default();
        let strategy_config = StrategyConfig::default();
        let position_manager = PositionManager::new(strategy_config.clone());
        let signal_generator = SignalGenerator::new(strategy_config);

        let mut engine = BacktestEngine::new(backtest_config, position_manager, signal_generator);

        // 创建模拟数据
        let mut market_data = Vec::new();
        let mut predictions = Vec::new();

        for i in 0..100 {
            market_data.push(MarketData {
                timestamp: Utc::now(),
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.0 + i as f64,
                volume: 1000.0,
            });

            predictions.push(Prediction {
                timestamp: Utc::now(),
                price_change: if i % 2 == 0 { 0.01 } else { -0.01 },
                signal: TradingSignal::Hold,
                confidence: 0.7,
            });
        }

        let result = engine.run(&market_data, &predictions);
        assert!(result.final_capital > 0.0);
    }
}
