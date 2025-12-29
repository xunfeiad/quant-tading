use std::collections::HashMap;

use crate::types::okx::channel::ChannelType;
use crate::types::okx::{Args, RequestMessage, ResponseMessage};
use crate::{
    Credential, ExchangeError, ExchangeResult, ExchangeTrait, SubscribeChannel,
    types::okx::{LoginArgs, Operation, channel::Channel},
};
use async_trait::async_trait;
use bytes::Bytes;
use flume::Sender;
use futures_util::{SinkExt, StreamExt};
use reqwest::Method;
use tokio_tungstenite::tungstenite::Utf8Bytes;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, tungstenite::protocol::Message};
use tracing::{debug, error, warn};

type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

pub type ChannelWsStream = HashMap<ChannelType, WsStream>;

/// Currently, Only support [MarkPariceChannel].
pub struct Okx {
    pub credential: Credential,
    /// The channel need to subscribe
    pub args: Vec<Args>,
    /// Channel's ws connection
    pub ws_streams: ChannelWsStream,
    /// Sender
    pub pipline_sender: Sender<ResponseMessage>,
}

impl Okx {
    fn group_channel_by_channel_type(&self) -> HashMap<ChannelType, Vec<Args>> {
        let mut channel_mapping: HashMap<ChannelType, Vec<Args>> = HashMap::new();
        for arg in self.args.clone().into_iter() {
            channel_mapping
                .entry(arg.channel.channel_type())
                .or_insert_with(Vec::new)
                .push(arg.clone());
        }
        channel_mapping
    }

    async fn handle_message(
        mut ws_stream: WsStream,
        sender: Sender<ResponseMessage>,
    ) -> ExchangeResult<()> {
        let (mut ws_writer, mut ws_reader) = ws_stream.split();
        let resp = match ws_reader.next().await {
            Some(Ok(message)) => {
                match message {
                    Message::Text(text) => serde_json::from_str(&text).map_err(|e| {
                        ExchangeError::Custom(format!("Failed to parse message: {}", e))
                    }),
                    Message::Pong(_) => {
                        debug!("Received pong");
                        Ok(ResponseMessage {
                            event: Some(Operation::Pong),
                            code: Some("0".to_string()),
                            // conn_id: Some(self.connection_id.clone()),
                            ..Default::default()
                        })
                    }
                    Message::Ping(_) => {
                        debug!("Received ping, sending pong");
                        // 对于ping消息，我们直接返回一个pong响应
                        ws_writer.send(Message::Pong(Bytes::new())).await?;
                        Ok(ResponseMessage {
                            event: Some(Operation::Pong),
                            code: Some("0".to_string()),
                            ..Default::default()
                        })
                    }
                    Message::Close(_) => {
                        warn!("WebSocket connection closed by server");
                        Err(ExchangeError::Custom("WebSocket connection closed".into()))
                    }
                    _ => Err(ExchangeError::Custom("Unsupported message type".into())),
                }
            }
            Some(Err(e)) => Err(ExchangeError::Custom(format!("WebSocket error: {}", e))),
            None => Err(ExchangeError::Custom("WebSocket connection closed".into())),
        };

        sender.send_async(resp?).await?;
        Ok(())
    }
}

#[async_trait]
impl SubscribeChannel for Okx {
    fn exchange() -> String {
        "Okx".to_string()
    }
    async fn subscribe_all(&mut self) -> ExchangeResult<()> {
        let need_login = self.args.iter().any(|arg| arg.channel.need_auth());

        if need_login {
            self.login().await?;
        }

        for (channel_type, channels) in self.group_channel_by_channel_type().into_iter() {
            let request_message =
                RequestMessage { id: None, op: Operation::Subscribe, args: channels };
            match serde_json::to_string(&request_message) {
                Ok(message) => {
                    if let Some(ws_stream) = self.ws_streams.get_mut(&channel_type) {
                        ws_stream.send(Message::Text(Utf8Bytes::from(message))).await?;
                    }
                }
                Err(e) => error!("Serialized request message failed with {e:?}"),
            }
        }
        Ok(())
    }

    async fn unsubscribe(&mut self, channels: Vec<Channel>) -> ExchangeResult<()> {
        for channel in channels.into_iter() {
            let channel_type = channel.channel_type();
            let request_message =
                RequestMessage { id: None, op: Operation::Unsubscribe, args: vec![channel] };
            match serde_json::to_string(&request_message) {
                Ok(message) => {
                    if let Some(ws_stream) = self.ws_streams.get_mut(&channel_type) {
                        ws_stream.send(Message::Text(Utf8Bytes::from(message))).await?;
                    }
                }
                Err(e) => error!("Serialized request message failed with {e:?}"),
            }
        }
        Ok(())
    }
}

#[async_trait]
impl ExchangeTrait for Okx {
    async fn login(&mut self) -> ExchangeResult<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ExchangeError::Custom(e.to_string()))?
            .as_secs_f64();

        let sign = self.gen_signature(
            &Method::GET,
            "/users/self/verify",
            "",
            timestamp,
            &self.credential.key_secret,
        )?;

        let login_args = LoginArgs {
            api_key: self.credential.api_key.clone(),
            passphrase: self.credential.clone().passphrase.unwrap(),
            timestamp,
            sign,
        };
        let id: u64 = rand::random();
        let request_message =
            RequestMessage { id: Some(id), op: Operation::Login, args: vec![login_args] };

        if let Some(ws_sender) = self.ws_streams.get_mut(&ChannelType::Private) {
            ws_sender
                .send(Message::Text(Utf8Bytes::from(serde_json::to_string(&request_message)?)))
                .await?;
        }
        Ok(())
    }

    async fn handle_channel_message(mut self) -> ExchangeResult<()> {
        let mut handlers = Vec::new();
        let ws_streams = std::mem::take(&mut self.ws_streams);

        for (_, ws_stream) in ws_streams {
            let sender = self.pipline_sender.clone();
            let handler = tokio::spawn(async move {
                if let Err(e) = Self::handle_message(ws_stream, sender).await {
                    error!("Error occured: {e:?}")
                }
            });
            handlers.push(handler);
        }

        let _ = futures_util::future::join_all(handlers);
        Ok(())
    }
}
