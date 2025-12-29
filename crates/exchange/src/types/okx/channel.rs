use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Channel {
    CommonChannel(CommonChannel),
    MarkPriceCandle(MarkPriceCandle),
    IndexCandle(IndexCandle),
    SpreadChannel(SpreadChannel),
    SpreadCandle(SpreadCandle),
}

impl Channel {
    pub fn need_auth(&self) -> bool {
        match self.channel_type() {
            ChannelType::Business | ChannelType::Private => true,
            ChannelType::Public => false,
        }
    }

    pub fn channel_type(&self) -> ChannelType {
        match self {
            Channel::CommonChannel(common_channel) => match common_channel {
                CommonChannel::SprdPublicTrades | CommonChannel::SprdTickers => {
                    ChannelType::Business
                }
                _ => ChannelType::Public,
            },
            Channel::MarkPriceCandle(_)
            | Channel::IndexCandle(_)
            | Channel::SpreadChannel(_)
            | Channel::SpreadCandle(_) => ChannelType::Business,
        }
    }

    pub fn channel_url(&self, base_url: String) -> String {
        match self.channel_type() {
            ChannelType::Business => base_url + "/ws/v5/business",
            ChannelType::Private => base_url + "/ws/v5/private",
            ChannelType::Public => base_url + "/ws/v5/public",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CommonChannel {
    #[default]
    FundingRate,
    PriceLimit,
    MarkPrice,
    IndexTickers,
    SprdPublicTrades,
    SprdTickers,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "kebab-case")]
pub enum SpreadChannel {
    #[default]
    SprdBboTbt,
    SprdBooks5,
    SprdBooksL2Tbt,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkPriceCandle {
    // 常规时间频道
    #[serde(rename = "mark-price-candle3M")]
    MarkPriceCandle3M,
    #[serde(rename = "mark-price-candle1M")]
    MarkPriceCandle1M,
    #[serde(rename = "mark-price-candle1W")]
    MarkPriceCandle1W,
    #[serde(rename = "mark-price-candle1D")]
    MarkPriceCandle1D,
    #[serde(rename = "mark-price-candle2D")]
    MarkPriceCandle2D,
    #[serde(rename = "mark-price-candle3D")]
    MarkPriceCandle3D,
    #[serde(rename = "mark-price-candle5D")]
    MarkPriceCandle5D,
    #[serde(rename = "mark-price-candle12H")]
    MarkPriceCandle12H,
    #[serde(rename = "mark-price-candle6H")]
    MarkPriceCandle6H,
    #[serde(rename = "mark-price-candle4H")]
    MarkPriceCandle4H,
    #[serde(rename = "mark-price-candle2H")]
    MarkPriceCandle2H,
    #[serde(rename = "mark-price-candle1H")]
    MarkPriceCandle1H,
    #[serde(rename = "mark-price-candle30m")]
    MarkPriceCandle30m,
    #[serde(rename = "mark-price-candle15m")]
    MarkPriceCandle15m,
    #[serde(rename = "mark-price-candle5m")]
    MarkPriceCandle5m,
    #[serde(rename = "mark-price-candle3m")]
    MarkPriceCandle3m,
    #[serde(rename = "mark-price-candle1m")]
    MarkPriceCandle1m,

    // UTC时间频道
    #[serde(rename = "mark-price-candle3Mutc")]
    MarkPriceCandle3MUtc,
    #[serde(rename = "mark-price-candle1Mutc")]
    MarkPriceCandle1MUtc,
    #[serde(rename = "mark-price-candle1Wutc")]
    MarkPriceCandle1WUtc,
    #[serde(rename = "mark-price-candle1Dutc")]
    MarkPriceCandle1DUtc,
    #[serde(rename = "mark-price-candle2Dutc")]
    MarkPriceCandle2DUtc,
    #[serde(rename = "mark-price-candle3Dutc")]
    MarkPriceCandle3DUtc,
    #[serde(rename = "mark-price-candle5Dutc")]
    MarkPriceCandle5DUtc,
    #[serde(rename = "mark-price-candle12Hutc")]
    MarkPriceCandle12HUtc,
    #[serde(rename = "mark-price-candle6Hutc")]
    MarkPriceCandle6HUtc,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexCandle {
    // 常规时间频道
    #[serde(rename = "index-candle3M")]
    IndexCandle3M,
    #[serde(rename = "index-candle1M")]
    IndexCandle1M,
    #[serde(rename = "index-candle1W")]
    IndexCandle1W,
    #[serde(rename = "index-candle1D")]
    IndexCandle1D,
    #[serde(rename = "index-candle2D")]
    IndexCandle2D,
    #[serde(rename = "index-candle3D")]
    IndexCandle3D,
    #[serde(rename = "index-candle5D")]
    IndexCandle5D,
    #[serde(rename = "index-candle12H")]
    IndexCandle12H,
    #[serde(rename = "index-candle6H")]
    IndexCandle6H,
    #[serde(rename = "index-candle4H")]
    IndexCandle4H,
    #[serde(rename = "index-candle2H")]
    IndexCandle2H,
    #[serde(rename = "index-candle1H")]
    IndexCandle1H,
    #[serde(rename = "index-candle30m")]
    IndexCandle30m,
    #[serde(rename = "index-candle15m")]
    IndexCandle15m,
    #[serde(rename = "index-candle5m")]
    IndexCandle5m,
    #[serde(rename = "index-candle3m")]
    IndexCandle3m,
    #[serde(rename = "index-candle1m")]
    IndexCandle1m,

    // UTC时间频道
    #[serde(rename = "index-candle3Mutc")]
    IndexCandle3MUtc,
    #[serde(rename = "index-candle1Mutc")]
    IndexCandle1MUtc,
    #[serde(rename = "index-candle1Wutc")]
    IndexCandle1WUtc,
    #[serde(rename = "index-candle1Dutc")]
    IndexCandle1DUtc,
    #[serde(rename = "index-candle2Dutc")]
    IndexCandle2DUtc,
    #[serde(rename = "index-candle3Dutc")]
    IndexCandle3DUtc,
    #[serde(rename = "index-candle5Dutc")]
    IndexCandle5DUtc,
    #[serde(rename = "index-candle12Hutc")]
    IndexCandle12HUtc,
    #[serde(rename = "index-candle6Hutc")]
    IndexCandle6HUtc,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelType {
    Public,
    Private,
    Business,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpreadCandle {
    // 常规时间频道
    #[serde(rename = "sprd-candle3M")]
    SprdCandle3M,
    #[serde(rename = "sprd-candle1M")]
    SprdCandle1M,
    #[serde(rename = "sprd-candle1W")]
    SprdCandle1W,
    #[serde(rename = "sprd-candle1D")]
    SprdCandle1D,
    #[serde(rename = "sprd-candle2D")]
    SprdCandle2D,
    #[serde(rename = "sprd-candle3D")]
    SprdCandle3D,
    #[serde(rename = "sprd-candle5D")]
    SprdCandle5D,
    #[serde(rename = "sprd-candle12H")]
    SprdCandle12H,
    #[serde(rename = "sprd-candle6H")]
    SprdCandle6H,
    #[serde(rename = "sprd-candle4H")]
    SprdCandle4H,
    #[serde(rename = "sprd-candle2H")]
    SprdCandle2H,
    #[serde(rename = "sprd-candle1H")]
    SprdCandle1H,
    #[serde(rename = "sprd-candle30m")]
    SprdCandle30m,
    #[serde(rename = "sprd-candle15m")]
    SprdCandle15m,
    #[serde(rename = "sprd-candle5m")]
    SprdCandle5m,
    #[serde(rename = "sprd-candle3m")]
    SprdCandle3m,
    #[serde(rename = "sprd-candle1m")]
    SprdCandle1m,

    // UTC时间频道
    #[serde(rename = "sprd-candle3Mutc")]
    SprdCandle3MUtc,
    #[serde(rename = "sprd-candle1Mutc")]
    SprdCandle1MUtc,
    #[serde(rename = "sprd-candle1Wutc")]
    SprdCandle1WUtc,
    #[serde(rename = "sprd-candle1Dutc")]
    SprdCandle1DUtc,
    #[serde(rename = "sprd-candle2Dutc")]
    SprdCandle2DUtc,
    #[serde(rename = "sprd-candle3Dutc")]
    SprdCandle3DUtc,
    #[serde(rename = "sprd-candle5Dutc")]
    SprdCandle5DUtc,
    #[serde(rename = "sprd-candle12Hutc")]
    SprdCandle12HUtc,
    #[serde(rename = "sprd-candle6Hutc")]
    SprdCandle6HUtc,
}
