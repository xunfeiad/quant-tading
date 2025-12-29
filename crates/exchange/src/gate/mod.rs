mod spot;
mod withdraw;

use crate::Credential;
use reqwest::Client;

pub const BASE_URL: &str = "https://api.gateio.ws/api/v4";

pub struct GateApi<'a> {
    credential: Credential,
    client: &'a Client,
}
