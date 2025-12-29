mod balance;
// mod gate;
mod helpers;
mod okx;
pub mod types;

use std::collections::HashMap;

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose};
use hmac::{Hmac, Mac};
use reqwest::Method;
use thiserror::Error;

use crate::{
    okx::Okx,
    types::okx::{ResponseMessage, channel::Channel},
};
pub type ExchangeResult<T> = Result<T, ExchangeError>;

#[derive(Debug, Error)]
pub enum ExchangeError {
    #[error("Failed to generate signature: {0}")]
    GenSign(String),
    #[error("Failed to send request: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("{0}")]
    Custom(String),
    #[error("{0}")]
    Weboscket(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("{0}")]
    Serde(#[from] serde_json::Error),

    #[error(transparent)]
    Flume(#[from] flume::SendError<ResponseMessage>),
}

impl From<String> for ExchangeError {
    fn from(value: String) -> Self {
        ExchangeError::Custom(value)
    }
}

#[derive(Clone)]
pub struct Credential {
    pub api_key: String,
    pub key_secret: String,
    pub passphrase: Option<String>,
}

#[async_trait]
pub trait SubscribeChannel {
    fn exchange() -> String;

    async fn subscribe_all(&mut self) -> ExchangeResult<()>;

    async fn unsubscribe(&mut self, channels: Vec<Channel>) -> ExchangeResult<()>;
}

#[async_trait]
pub trait ExchangeTrait {
    fn gen_signature(
        &self,
        method: &Method,
        path: &str,
        body: &str,
        timestamp: f64,
        secret: &str,
    ) -> ExchangeResult<String> {
        let mut mac = Hmac::<sha2::Sha256>::new_from_slice(secret.as_bytes())
            .map_err(|e| ExchangeError::GenSign(e.to_string()))?;
        let message = format!("{}{}{}{}", timestamp, method.as_str(), path, body);
        mac.update(message.as_bytes());

        let result = mac.finalize();
        Ok(general_purpose::STANDARD.encode(result.into_bytes()))
    }

    async fn login(&mut self) -> ExchangeResult<()>;

    async fn handle_channel_message(mut self) -> ExchangeResult<()>;
}

pub async fn init_okx(
    channels: Vec<Channel>,
    base_url: String,
    credential: Credential,
) -> ExchangeResult<()> {
    let mut ws_streams = HashMap::new();
    for channel in channels.iter() {
        let channel_type = channel.channel_type();
        let channel_url = channel.channel_url(base_url.clone());
        println!("{:?}", channel_url);
        let (ws_stream, _) = tokio_tungstenite::connect_async(channel_url).await?;
        ws_streams.insert(channel_type, ws_stream);
    }
    let (sender, receiver) = flume::unbounded();
    let mut okx =
        Okx { credential, subscribe_channel: channels, ws_streams, pipline_sender: sender };
    okx.subscribe_all().await?;
    okx.handle_channel_message().await?;
    while let Ok(msg) = receiver.recv_async().await {
        println!("{:?}", msg);
    }
    Ok(())
}
