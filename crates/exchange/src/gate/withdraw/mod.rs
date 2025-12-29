use crate::{Credential, ExchangeResult};
pub mod types;
use crate::gate::{BASE_URL, GateApi};
use tracing::{debug, info};
use types::*;

impl GateApi<'_> {
    pub async fn withdrawal(&self, body: WithdrawalBody) -> ExchangeResult<WithdrawalResponse> {
        let url = format!("/withdrawals");
        let full_url = format!("{}{}", BASE_URL, url);
        let sign_headers =
            crate::helpers::gen_sign("POST", &full_url, None, Some(&body), &self.credential)?;
        let response = self
            .client
            .post(&url)
            .json(&body)
            .headers(sign_headers)
            .send()
            .await?;

        let response = response.error_for_status()?;

        debug!("Withdrawal response: {:?}", &response);

        Ok(response.json().await?)
    }

    pub async fn withdrawal_with_uid(
        &self,
        body: WithdrawalPushBody,
    ) -> ExchangeResult<WithdrawalPushResponse> {
        let url = format!("/withdrawals/push");
        let full_url = format!("{}{}", BASE_URL, url);
        let sign_headers =
            crate::helpers::gen_sign("POST", &full_url, None, Some(&body), &self.credential)?;
        let response = self
            .client
            .post(&url)
            .json(&body)
            .headers(sign_headers)
            .send()
            .await?;

        let response = response.error_for_status()?;

        debug!("Withdrawal with id response: {:?}", &response);

        Ok(response.json().await?)
    }

    pub async fn cancel_withdrawal(&self, id: i64) -> ExchangeResult<CancelWithdrawalResponse> {
        let url = format!("/withdrawals/{}", id);
        let full_url = format!("{}{}", BASE_URL, url);
        let sign_headers = crate::helpers::gen_sign(
            "POST",
            &full_url,
            None,
            Option::<String>::None,
            &self.credential,
        )?;
        let response = self
            .client
            .delete(&url)
            .headers(sign_headers)
            .send()
            .await?;

        let response = response.error_for_status()?;

        debug!("Cancel withdrawal with response: {:?}", &response);
        Ok(response.json().await?)
    }
}
