use crate::{Credential, ExchangeError, ExchangeResult};
use hmac::{Hmac, Mac};
use reqwest::{
    Method,
    header::{HeaderMap, HeaderValue},
};
use serde::Serialize;
use sha2::{Digest, Sha512};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha512 = Hmac<Sha512>;

pub struct Signature {
    pub key: String,
    pub timestamp: f64,
    pub sign: String,
}

pub fn gen_sign<'a>(
    method: &Method,
    url: &str,
    query_string: Option<&str>,
    payload: Option<impl Serialize>,
    credential: &Credential,
) -> ExchangeResult<HeaderMap> {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();

    let mut hasher = Sha512::new();

    let payload_str = match payload {
        Some(payload) => serde_json::to_string(&payload).unwrap_or_default(),
        None => "".to_string(),
    };
    hasher.update(payload_str.as_bytes());
    let hashed_payload = hex::encode(hasher.finalize());

    let s = format!(
        "{}\n{}\n{}\n{}\n{}",
        method.as_str(),
        url,
        query_string.unwrap_or(""),
        hashed_payload,
        t
    );
    let mut mac = HmacSha512::new_from_slice(credential.key_secret.as_bytes())
        .map_err(|e| ExchangeError::GenSign(e.to_string()))?;
    mac.update(s.as_bytes());
    let sign = hex::encode(mac.finalize().into_bytes());
    let mut headers = HeaderMap::new();
    headers.insert(
        "KEY",
        credential
            .api_key
            .parse::<HeaderValue>()
            .map_err(|e| ExchangeError::GenSign(e.to_string()))?,
    );
    headers.insert(
        "Timestamp",
        t.to_string().parse::<HeaderValue>().map_err(|e| ExchangeError::GenSign(e.to_string()))?,
    );
    headers
        .insert(
            "SIGN",
            sign.parse::<HeaderValue>().map_err(|e| ExchangeError::GenSign(e.to_string()))?,
        )
        .unwrap();
    Ok(headers)
}
