//! 深度学习模型（LSTM、GRU、Transformer）
//!
//! 使用 tch-rs (PyTorch bindings for Rust) 实现

use crate::models::TimeSeriesModel;
use crate::types::{MLError, MLResult};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

/// LSTM 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    pub input_size: i64,
    pub hidden_size: i64,
    pub num_layers: i64,
    pub output_size: i64,
    pub dropout: f64,
    pub learning_rate: f64,
    pub epochs: i64,
    pub batch_size: i64,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            input_size: 20,
            hidden_size: 128,
            num_layers: 2,
            output_size: 1,
            dropout: 0.2,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
        }
    }
}

/// LSTM 网络
#[derive(Debug)]
struct LSTMNet {
    lstm: nn::LSTM,
    linear: nn::Linear,
    dropout: f64,
}

impl LSTMNet {
    fn new(vs: &nn::Path, config: &LSTMConfig) -> Self {
        let lstm_config = nn::RNNConfig {
            dropout: config.dropout,
            num_layers: config.num_layers,
            ..Default::default()
        };

        let lstm = nn::lstm(
            vs,
            config.input_size,
            config.hidden_size,
            lstm_config,
        );

        let linear = nn::linear(
            vs / "linear",
            config.hidden_size,
            config.output_size,
            Default::default(),
        );

        Self {
            lstm,
            linear,
            dropout: config.dropout,
        }
    }

    fn forward(&self, input: &Tensor, train: bool) -> Tensor {
        // input shape: [batch_size, seq_len, input_size]
        let (lstm_out, _) = self.lstm.seq(input);

        // 取最后一个时间步的输出
        let last_output = lstm_out.select(1, -1);

        // 应用 dropout
        let dropped = if train && self.dropout > 0.0 {
            last_output.dropout(self.dropout, true)
        } else {
            last_output
        };

        // 全连接层
        self.linear.forward(&dropped)
    }
}

/// LSTM 模型
pub struct LSTMModel {
    config: LSTMConfig,
    vs: nn::VarStore,
    net: Option<LSTMNet>,
    device: Device,
}

impl LSTMModel {
    pub fn new(config: LSTMConfig) -> Self {
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let vs = nn::VarStore::new(device);

        Self {
            config,
            vs,
            net: None,
            device,
        }
    }

    fn array3_to_tensor(&self, arr: &Array3<f64>) -> MLResult<Tensor> {
        let shape = arr.shape();
        let data: Vec<f64> = arr.iter().cloned().collect();

        Ok(Tensor::from_slice(&data)
            .view([shape[0] as i64, shape[1] as i64, shape[2] as i64])
            .to_device(self.device))
    }

    fn array2_to_tensor(&self, arr: &Array2<f64>) -> MLResult<Tensor> {
        let shape = arr.shape();
        let data: Vec<f64> = arr.iter().cloned().collect();

        Ok(Tensor::from_slice(&data)
            .view([shape[0] as i64, shape[1] as i64])
            .to_device(self.device))
    }

    fn tensor_to_array2(&self, tensor: &Tensor) -> MLResult<Array2<f64>> {
        let shape: Vec<i64> = tensor.size();
        let data: Vec<f64> = Vec::try_from(tensor.to_device(Device::Cpu))
            .map_err(|e| MLError::Torch(format!("张量转换失败: {:?}", e)))?;

        Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data)
            .map_err(|e| MLError::Torch(format!("形状转换失败: {:?}", e)))
    }
}

#[async_trait]
impl TimeSeriesModel for LSTMModel {
    async fn train_sequence(
        &mut self,
        x_train: &Array3<f64>,
        y_train: &Array2<f64>,
    ) -> MLResult<()> {
        // 初始化网络
        let net = LSTMNet::new(&self.vs.root(), &self.config);
        let mut opt = nn::Adam::default().build(&self.vs, self.config.learning_rate)
            .map_err(|e| MLError::Training(format!("优化器创建失败: {:?}", e)))?;

        // 转换数据为张量
        let x_tensor = self.array3_to_tensor(x_train)?;
        let y_tensor = self.array2_to_tensor(y_train)?;

        let n_samples = x_train.shape()[0] as i64;
        let n_batches = (n_samples + self.config.batch_size - 1) / self.config.batch_size;

        // 训练循环
        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0;

            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * self.config.batch_size;
                let end_idx = ((batch_idx + 1) * self.config.batch_size).min(n_samples);

                let batch_x = x_tensor.narrow(0, start_idx, end_idx - start_idx);
                let batch_y = y_tensor.narrow(0, start_idx, end_idx - start_idx);

                let predictions = net.forward(&batch_x, true);
                let loss = predictions.mse_loss(&batch_y, tch::Reduction::Mean);

                opt.backward_step(&loss);
                total_loss += f64::try_from(&loss)
                    .map_err(|e| MLError::Training(format!("损失转换失败: {:?}", e)))?;
            }

            let avg_loss = total_loss / n_batches as f64;
            if (epoch + 1) % 10 == 0 {
                tracing::info!("Epoch {}/{}, Loss: {:.6}", epoch + 1, self.config.epochs, avg_loss);
            }
        }

        self.net = Some(net);
        Ok(())
    }

    async fn predict_sequence(&self, x: &Array3<f64>) -> MLResult<Array2<f64>> {
        let net = self.net.as_ref()
            .ok_or_else(|| MLError::Prediction("模型未训练".to_string()))?;

        let x_tensor = self.array3_to_tensor(x)?;

        let predictions = tch::no_grad(|| {
            net.forward(&x_tensor, false)
        });

        self.tensor_to_array2(&predictions)
    }

    async fn save(&self, path: &str) -> MLResult<()> {
        self.vs.save(path)
            .map_err(|e| MLError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        let config_path = format!("{}.config", path);
        let config_json = serde_json::to_string(&self.config)
            .map_err(|e| MLError::Serialization(e.to_string()))?;
        std::fs::write(config_path, config_json)?;

        Ok(())
    }

    async fn load(path: &str) -> MLResult<Self> {
        let config_path = format!("{}.config", path);
        let config_json = std::fs::read_to_string(config_path)?;
        let config: LSTMConfig = serde_json::from_str(&config_json)
            .map_err(|e| MLError::Serialization(e.to_string()))?;

        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let mut vs = nn::VarStore::new(device);
        let net = LSTMNet::new(&vs.root(), &config);

        vs.load(path)
            .map_err(|e| MLError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        Ok(Self {
            config,
            vs,
            net: Some(net),
            device,
        })
    }
}

/// GRU 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRUConfig {
    pub input_size: i64,
    pub hidden_size: i64,
    pub num_layers: i64,
    pub output_size: i64,
    pub dropout: f64,
    pub learning_rate: f64,
    pub epochs: i64,
    pub batch_size: i64,
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            input_size: 20,
            hidden_size: 128,
            num_layers: 2,
            output_size: 1,
            dropout: 0.2,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
        }
    }
}

/// GRU 网络
#[derive(Debug)]
struct GRUNet {
    gru: nn::GRU,
    linear: nn::Linear,
    dropout: f64,
}

impl GRUNet {
    fn new(vs: &nn::Path, config: &GRUConfig) -> Self {
        let gru_config = nn::RNNConfig {
            dropout: config.dropout,
            num_layers: config.num_layers,
            ..Default::default()
        };

        let gru = nn::gru(
            vs,
            config.input_size,
            config.hidden_size,
            gru_config,
        );

        let linear = nn::linear(
            vs / "linear",
            config.hidden_size,
            config.output_size,
            Default::default(),
        );

        Self {
            gru,
            linear,
            dropout: config.dropout,
        }
    }

    fn forward(&self, input: &Tensor, train: bool) -> Tensor {
        let (gru_out, _) = self.gru.seq(input);
        let last_output = gru_out.select(1, -1);

        let dropped = if train && self.dropout > 0.0 {
            last_output.dropout(self.dropout, true)
        } else {
            last_output
        };

        self.linear.forward(&dropped)
    }
}

/// GRU 模型
pub struct GRUModel {
    config: GRUConfig,
    vs: nn::VarStore,
    net: Option<GRUNet>,
    device: Device,
}

impl GRUModel {
    pub fn new(config: GRUConfig) -> Self {
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let vs = nn::VarStore::new(device);

        Self {
            config,
            vs,
            net: None,
            device,
        }
    }

    fn array3_to_tensor(&self, arr: &Array3<f64>) -> MLResult<Tensor> {
        let shape = arr.shape();
        let data: Vec<f64> = arr.iter().cloned().collect();

        Ok(Tensor::from_slice(&data)
            .view([shape[0] as i64, shape[1] as i64, shape[2] as i64])
            .to_device(self.device))
    }

    fn array2_to_tensor(&self, arr: &Array2<f64>) -> MLResult<Tensor> {
        let shape = arr.shape();
        let data: Vec<f64> = arr.iter().cloned().collect();

        Ok(Tensor::from_slice(&data)
            .view([shape[0] as i64, shape[1] as i64])
            .to_device(self.device))
    }

    fn tensor_to_array2(&self, tensor: &Tensor) -> MLResult<Array2<f64>> {
        let shape: Vec<i64> = tensor.size();
        let data: Vec<f64> = Vec::try_from(tensor.to_device(Device::Cpu))
            .map_err(|e| MLError::Torch(format!("张量转换失败: {:?}", e)))?;

        Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data)
            .map_err(|e| MLError::Torch(format!("形状转换失败: {:?}", e)))
    }
}

#[async_trait]
impl TimeSeriesModel for GRUModel {
    async fn train_sequence(
        &mut self,
        x_train: &Array3<f64>,
        y_train: &Array2<f64>,
    ) -> MLResult<()> {
        let net = GRUNet::new(&self.vs.root(), &self.config);
        let mut opt = nn::Adam::default().build(&self.vs, self.config.learning_rate)
            .map_err(|e| MLError::Training(format!("优化器创建失败: {:?}", e)))?;

        let x_tensor = self.array3_to_tensor(x_train)?;
        let y_tensor = self.array2_to_tensor(y_train)?;

        let n_samples = x_train.shape()[0] as i64;
        let n_batches = (n_samples + self.config.batch_size - 1) / self.config.batch_size;

        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0;

            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * self.config.batch_size;
                let end_idx = ((batch_idx + 1) * self.config.batch_size).min(n_samples);

                let batch_x = x_tensor.narrow(0, start_idx, end_idx - start_idx);
                let batch_y = y_tensor.narrow(0, start_idx, end_idx - start_idx);

                let predictions = net.forward(&batch_x, true);
                let loss = predictions.mse_loss(&batch_y, tch::Reduction::Mean);

                opt.backward_step(&loss);
                total_loss += f64::try_from(&loss)
                    .map_err(|e| MLError::Training(format!("损失转换失败: {:?}", e)))?;
            }

            let avg_loss = total_loss / n_batches as f64;
            if (epoch + 1) % 10 == 0 {
                tracing::info!("Epoch {}/{}, Loss: {:.6}", epoch + 1, self.config.epochs, avg_loss);
            }
        }

        self.net = Some(net);
        Ok(())
    }

    async fn predict_sequence(&self, x: &Array3<f64>) -> MLResult<Array2<f64>> {
        let net = self.net.as_ref()
            .ok_or_else(|| MLError::Prediction("模型未训练".to_string()))?;

        let x_tensor = self.array3_to_tensor(x)?;

        let predictions = tch::no_grad(|| {
            net.forward(&x_tensor, false)
        });

        self.tensor_to_array2(&predictions)
    }

    async fn save(&self, path: &str) -> MLResult<()> {
        self.vs.save(path)
            .map_err(|e| MLError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        let config_path = format!("{}.config", path);
        let config_json = serde_json::to_string(&self.config)
            .map_err(|e| MLError::Serialization(e.to_string()))?;
        std::fs::write(config_path, config_json)?;

        Ok(())
    }

    async fn load(path: &str) -> MLResult<Self> {
        let config_path = format!("{}.config", path);
        let config_json = std::fs::read_to_string(config_path)?;
        let config: GRUConfig = serde_json::from_str(&config_json)
            .map_err(|e| MLError::Serialization(e.to_string()))?;

        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let mut vs = nn::VarStore::new(device);
        let net = GRUNet::new(&vs.root(), &config);

        vs.load(path)
            .map_err(|e| MLError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        Ok(Self {
            config,
            vs,
            net: Some(net),
            device,
        })
    }
}
