use serde::Serialize;

#[derive(Serialize)]
pub struct CandleTicks {
    pub currency_pair: String,
    pub limit: Option<u32>,
    pub from: Option<i64>,
    pub to: Option<i64>,
    pub interval: Option<String>,
}
