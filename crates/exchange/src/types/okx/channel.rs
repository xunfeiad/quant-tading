use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Channel {
    CommonChannel(CommonChannel),
    MarkPriceCandle(MarkPriceCandle),
    IndexCandle(IndexCandle),
}

impl Channel {
    pub fn need_auth(&self) -> bool {
        match self {
            Channel::CommonChannel(common_channel) => match common_channel {
                CommonChannel::FundingRate => false,
                CommonChannel::PriceLimit => false,
                CommonChannel::MarkPrice => false,
                CommonChannel::IndexTickers => false,
                CommonChannel::SprdPublicTrades => true,
            },
            Channel::MarkPriceCandle(_) => true,
            Channel::IndexCandle(_) => true,
        }
    }

    pub fn channel_type(&self) -> ChannelType {
        match self {
            Channel::CommonChannel(_) => ChannelType::Public,
            Channel::MarkPriceCandle(_) => ChannelType::Business,
            Channel::IndexCandle(_) => ChannelType::Business,
        }
    }

    pub fn channel_url(&self, base_url: String) -> String {
        match self {
            Channel::CommonChannel(_) => base_url + "/ws/v5/public",
            Channel::MarkPriceCandle(_) => base_url + "/ws/v5/business",
            Channel::IndexCandle(_) => base_url + "/ws/v5/business",
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
