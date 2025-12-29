use exchange::{
    Credential,
    types::okx::{Args, channel::Channel},
};
#[tokio::main]
async fn main() {
    let args = vec![Args {
        channel: Channel::MarkPriceCandle(
            exchange::types::okx::channel::MarkPriceCandle::MarkPriceCandle12H,
        ),
        inst_id: "BTC-USD-SWAP".to_string(),
        inst_type: None,
        inst_family: None,
    }];
    let base_url = "wss://ws.okx.com:8443".to_string();

    let credential = Credential {
        api_key: "79b084dc-3890-4387-a7e3-ca21a7bf0d33".to_string(),
        key_secret: "9408837542F2FAC8CD11F3478028433F".to_string(),
        passphrase: Some("Xf19941118!,,".to_string()),
    };
    exchange::init_okx(args, base_url, credential).await.unwrap();
}
