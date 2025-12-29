mod types;

use crate::gate::{BASE_URL, GateApi};

impl GateApi {
    fn spot_spec_currency(&self, currency: &str) -> Result<Vec<String>, Error> {
        let url = format!("{}/spot/currencies/{}", BASE_URL, currency);
        self.client.get(url)
    }
}
