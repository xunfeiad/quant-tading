pub mod channel;
use channel::Channel;
use serde::{Deserialize, Serialize};

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
pub struct Args {
    pub channel: Channel,
    #[serde(rename = "insType", skip_serializing_if = "Option::is_none")]
    pub inst_type: Option<InstrumentType>,
    #[serde(rename = "instFamily", skip_serializing_if = "Option::is_none")]
    pub inst_family: Option<String>,
    #[serde(rename = "instId")]
    pub inst_id: String,
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
