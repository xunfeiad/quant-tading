//! 交易策略生成器

use crate::types::{MLResult, Position, Prediction, TradingSignal, MarketData};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 策略配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// 买入阈值（预测值 > buy_threshold 时买入）
    pub buy_threshold: f64,
    /// 卖出阈值（预测值 < sell_threshold 时卖出）
    pub sell_threshold: f64,
    /// 最小置信度
    pub min_confidence: f64,
    /// 止损百分比
    pub stop_loss_pct: f64,
    /// 止盈百分比
    pub take_profit_pct: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            buy_threshold: 0.2,
            sell_threshold: -0.2,
            min_confidence: 0.6,
            stop_loss_pct: 0.02,      // 2% 止损
            take_profit_pct: 0.05,    // 5% 止盈
        }
    }
}

/// 交易信号生成器
pub struct SignalGenerator {
    config: StrategyConfig,
}

impl SignalGenerator {
    pub fn new(config: StrategyConfig) -> Self {
        Self { config }
    }

    /// 根据模型预测生成交易信号
    pub fn generate_signal(&self, prediction: &Prediction) -> TradingSignal {
        // 检查置信度
        if prediction.confidence < self.config.min_confidence {
            return TradingSignal::Hold;
        }

        // 根据预测值生成信号
        let value = prediction.price_change;

        if value > self.config.buy_threshold * 2.0 {
            TradingSignal::StrongBuy
        } else if value > self.config.buy_threshold {
            TradingSignal::Buy
        } else if value < self.config.sell_threshold * 2.0 {
            TradingSignal::StrongSell
        } else if value < self.config.sell_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// 批量生成交易信号
    pub fn generate_signals(&self, predictions: &[Prediction]) -> Vec<TradingSignal> {
        predictions
            .iter()
            .map(|pred| self.generate_signal(pred))
            .collect()
    }
}

/// 仓位管理器
#[derive(Debug, Clone)]
pub struct PositionManager {
    current_position: Position,
    entry_price: f64,
    stop_loss_price: f64,
    take_profit_price: f64,
    config: StrategyConfig,
}

impl PositionManager {
    pub fn new(config: StrategyConfig) -> Self {
        Self {
            current_position: Position::Neutral,
            entry_price: 0.0,
            stop_loss_price: 0.0,
            take_profit_price: 0.0,
            config,
        }
    }

    /// 获取当前持仓
    pub fn current_position(&self) -> Position {
        self.current_position
    }

    /// 开多仓
    pub fn open_long(&mut self, entry_price: f64) {
        self.current_position = Position::Long;
        self.entry_price = entry_price;
        self.stop_loss_price = entry_price * (1.0 - self.config.stop_loss_pct);
        self.take_profit_price = entry_price * (1.0 + self.config.take_profit_pct);
    }

    /// 开空仓
    pub fn open_short(&mut self, entry_price: f64) {
        self.current_position = Position::Short;
        self.entry_price = entry_price;
        self.stop_loss_price = entry_price * (1.0 + self.config.stop_loss_pct);
        self.take_profit_price = entry_price * (1.0 - self.config.take_profit_pct);
    }

    /// 平仓
    pub fn close_position(&mut self) {
        self.current_position = Position::Neutral;
        self.entry_price = 0.0;
        self.stop_loss_price = 0.0;
        self.take_profit_price = 0.0;
    }

    /// 检查是否触发止损或止盈
    pub fn check_exit(&self, current_price: f64) -> bool {
        match self.current_position {
            Position::Long => {
                current_price <= self.stop_loss_price || current_price >= self.take_profit_price
            }
            Position::Short => {
                current_price >= self.stop_loss_price || current_price <= self.take_profit_price
            }
            Position::Neutral => false,
        }
    }

    /// 根据信号更新持仓
    pub fn update_position(&mut self, signal: TradingSignal, current_price: f64) -> Option<TradeAction> {
        // 先检查是否需要止损/止盈
        if self.check_exit(current_price) {
            let action = TradeAction::Close {
                position: self.current_position,
                price: current_price,
            };
            self.close_position();
            return Some(action);
        }

        // 根据信号决定操作
        match (self.current_position, signal) {
            // 当前无持仓
            (Position::Neutral, TradingSignal::StrongBuy | TradingSignal::Buy) => {
                self.open_long(current_price);
                Some(TradeAction::OpenLong { price: current_price })
            }
            (Position::Neutral, TradingSignal::StrongSell | TradingSignal::Sell) => {
                self.open_short(current_price);
                Some(TradeAction::OpenShort { price: current_price })
            }

            // 当前多仓，收到卖出信号
            (Position::Long, TradingSignal::StrongSell | TradingSignal::Sell) => {
                let action = TradeAction::Close {
                    position: Position::Long,
                    price: current_price,
                };
                self.close_position();
                Some(action)
            }

            // 当前空仓，收到买入信号
            (Position::Short, TradingSignal::StrongBuy | TradingSignal::Buy) => {
                let action = TradeAction::Close {
                    position: Position::Short,
                    price: current_price,
                };
                self.close_position();
                Some(action)
            }

            // 其他情况：持有
            _ => None,
        }
    }

    /// 获取入场价格
    pub fn entry_price(&self) -> f64 {
        self.entry_price
    }

    /// 计算当前盈亏
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        match self.current_position {
            Position::Long => (current_price - self.entry_price) / self.entry_price,
            Position::Short => (self.entry_price - current_price) / self.entry_price,
            Position::Neutral => 0.0,
        }
    }
}

/// 交易动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    OpenLong { price: f64 },
    OpenShort { price: f64 },
    Close { position: Position, price: f64 },
}

/// 风险管理器
pub struct RiskManager {
    max_position_size: f64,
    max_drawdown_threshold: f64,
}

impl RiskManager {
    pub fn new(max_position_size: f64, max_drawdown_threshold: f64) -> Self {
        Self {
            max_position_size,
            max_drawdown_threshold,
        }
    }

    /// 计算仓位大小
    pub fn calculate_position_size(
        &self,
        account_balance: f64,
        confidence: f64,
    ) -> f64 {
        let base_size = account_balance * self.max_position_size;
        // 根据置信度调整仓位
        base_size * confidence
    }

    /// 检查是否触发最大回撤保护
    pub fn check_max_drawdown(&self, current_drawdown: f64) -> bool {
        current_drawdown >= self.max_drawdown_threshold
    }
}

/// 多策略组合
pub struct StrategyEnsemble {
    strategies: Vec<SignalGenerator>,
    weights: Vec<f64>,
}

impl StrategyEnsemble {
    pub fn new(strategies: Vec<SignalGenerator>, weights: Vec<f64>) -> MLResult<Self> {
        if strategies.len() != weights.len() {
            return Err(crate::types::MLError::InvalidConfig(
                "策略数量与权重数量不匹配".to_string()
            ));
        }

        let weight_sum: f64 = weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-6 {
            return Err(crate::types::MLError::InvalidConfig(
                "权重之和必须为 1".to_string()
            ));
        }

        Ok(Self { strategies, weights })
    }

    /// 生成组合信号（加权投票）
    pub fn generate_ensemble_signal(&self, prediction: &Prediction) -> TradingSignal {
        let mut vote_score = 0.0;

        for (strategy, &weight) in self.strategies.iter().zip(self.weights.iter()) {
            let signal = strategy.generate_signal(prediction);
            vote_score += signal.to_value() * weight;
        }

        TradingSignal::from_prediction(vote_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generator() {
        let config = StrategyConfig::default();
        let generator = SignalGenerator::new(config);

        let prediction = Prediction {
            timestamp: Utc::now(),
            price_change: 0.3,
            signal: TradingSignal::Buy,
            confidence: 0.8,
        };

        let signal = generator.generate_signal(&prediction);
        assert!(matches!(signal, TradingSignal::Buy | TradingSignal::StrongBuy));
    }

    #[test]
    fn test_position_manager() {
        let config = StrategyConfig::default();
        let mut manager = PositionManager::new(config);

        // 开多仓
        manager.open_long(100.0);
        assert_eq!(manager.current_position(), Position::Long);

        // 检查盈亏
        let pnl = manager.unrealized_pnl(105.0);
        assert!((pnl - 0.05).abs() < 1e-6);

        // 检查止损
        assert!(manager.check_exit(97.0)); // 低于止损价
        assert!(manager.check_exit(106.0)); // 高于止盈价
    }
}
