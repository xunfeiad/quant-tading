use exchange::{
    Credential,
    types::okx::channel::{Channel, CommonChannel},
};
#[tokio::main]
async fn main() {
    let channels = vec![Channel::CommonChannel(CommonChannel::FundingRate)];
    let base_url = "wss://ws.okx.com:8443".to_string();

    let credential = Credential {
        api_key: "79b084dc-3890-4387-a7e3-ca21a7bf0d33".to_string(),
        key_secret: "9408837542F2FAC8CD11F3478028433F".to_string(),
        passphrase: Some("Xf19941118!,,".to_string()),
    };
    exchange::init_okx(channels, base_url, credential).await.unwrap();
}
