pub mod channel;
use channel::Channel;
use serde::{Deserialize, Serialize};
use tracing::span;

use crate::types::okx::channel::MarkPriceCandle;

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginArgs {
    #[serde(rename = "apiKey")]
    pub api_key: String,
    pub passphrase: String,
    pub timestamp: f64,
    pub sign: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestMessage<T: Serialize> {
    pub id: Option<u64>,
    pub op: Operation,
    pub args: Vec<T>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    pub channel: Channel,
    #[serde(rename = "insType", skip_serializing_if = "Option::is_none")]
    pub inst_type: Option<InstrumentType>,
    #[serde(rename = "instFamily", skip_serializing_if = "Option::is_none")]
    pub inst_family: Option<String>,
    #[serde(rename = "instId", skip_serializing_if = "Option::is_none")]
    pub inst_id: Option<String>,
    #[serde(rename = "sprdId", skip_serializing_if = "Option::is_none")]
    pub sprd_id: Option<String>,
}

impl Args {
    pub fn new_mark_price(mark_price_candle: MarkPriceCandle, inst_id: String) -> Self {
        Self {
            channel: Channel::MarkPriceCandle(mark_price_candle),
            inst_type: Some(InstrumentType::SPOT),
            inst_id: Some(inst_id),
            inst_family: None,
            sprd_id: None,
        }
    }

    pub fn new_sprd_public_trades(sprd_id: String) -> Self {
        Self {
            channel: Channel::CommonChannel(channel::CommonChannel::SprdPublicTrades),
            inst_type: Some(InstrumentType::SPOT),
            inst_family: None,
            sprd_id: Some(sprd_id),
            inst_id: None,
        }
    }

    pub fn new_spread_channel(channel: channel::SpreadChannel, sprd_id: String) -> Self {
        Self {
            channel: Channel::SpreadChannel(channel),
            inst_type: None,
            inst_family: None,
            inst_id: None,
            sprd_id: Some(sprd_id),
        }
    }

    pub fn new_spread_ticker_channel(sprd_id: String) -> Self {
        Self {
            channel: Channel::CommonChannel(channel::CommonChannel::SprdPublicTrades),
            inst_type: None,
            inst_family: None,
            inst_id: None,
            sprd_id: Some(sprd_id),
        }
    }

    pub fn new_spread_candle_channel(
        spread_candle: channel::SpreadCandle,
        sprd_id: String,
    ) -> Self {
        Self {
            channel: Channel::SpreadCandle(spread_candle),
            inst_type: None,
            inst_family: None,
            inst_id: None,
            sprd_id: Some(sprd_id),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum InstrumentType {
    SPOT,
    MARGIN,
    SWAP,
    FUTURES,
    OPTION,
    ANY,
}

#[derive(Debug, Serialize, Default, Deserialize, Clone, PartialEq, Eq)]
pub enum Operation {
    #[default]
    #[serde(rename = "login")]
    Login,
    #[serde(rename = "subscribe")]
    Subscribe,
    #[serde(rename = "unsubscribe")]
    Unsubscribe,
    #[serde(rename = "error")]
    Error,

    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "channel-conn-count")]
    ChannelConnCount,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub id: Option<u64>,
    pub event: Option<Operation>,
    pub arg: Option<Args>,
    pub code: Option<String>,
    pub msg: Option<String>,
    #[serde(rename = "eventType")]
    pub event_type: Option<String>,
    #[serde(rename = "curPage")]
    pub cur_page: Option<i32>,
    #[serde(rename = "last_page")]
    pub last_page: Option<bool>,
    pub data: Option<serde_json::Value>,
    #[serde(rename = "connId")]
    pub conn_id: Option<String>,
}
