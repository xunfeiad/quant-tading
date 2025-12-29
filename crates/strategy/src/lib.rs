pub mod ema;

pub enum Signal {
    Buy,
    Sell,
    Hold,
}

pub trait Strategy {
    type Data;
    fn name() -> String;

    fn strategy(data: Self::Data) -> Signal;
}
