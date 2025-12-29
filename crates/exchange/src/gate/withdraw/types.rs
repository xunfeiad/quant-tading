use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct WithdrawalBody {
    /// Withdrawal order ID
    pub withdraw_order_id: Option<String>,
    /// Amount to withdraw
    pub amount: String,
    /// Currency to withdraw
    pub currency: String,
    /// Address to withdraw to
    pub address: Option<String>,
    /// Memo to withdraw to
    pub memo: Option<String>,
    pub withdraw_id: Option<String>,
    pub asset_class: Option<String>,
    pub chain: String,
}

#[derive(Debug, Serialize)]
pub struct WithdrawalPushBody {
    pub receive_uid: i64,
    pub currency: String,
    pub amount: String,
}

#[derive(Debug, Deserialize)]
pub struct WithdrawalResponse {
    /// Withdrawal order ID
    pub id: String,
    pub txid: String,
    pub withdraw_order_id: String,
    pub timestamp: String,
    pub amount: String,
    pub currency: String,
    pub address: String,
    pub memo: String,
    pub withdraw_id: String,
    pub asset_class: String,
    pub chain: String,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct WithdrawalPushResponse {
    pub id: i64,
}

#[derive(Debug, Deserialize)]
pub struct CancelWithdrawalResponse {
    pub id: i64,
    pub txid: String,
    pub timestamp: String,
    pub amount: String,
    pub currency: String,
    pub address: String,
    pub memo: String,
    pub block_number: String,
    pub chain: String,
    pub status: String,
}
