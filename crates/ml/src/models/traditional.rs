//! 传统机器学习模型

use crate::models::Model;
use crate::types::{MLError, MLResult};
use async_trait::async_trait;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 随机森林回归模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor {
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    trees: Vec<DecisionTree>,
}

impl RandomForestRegressor {
    pub fn new(n_trees: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            n_trees,
            max_depth,
            min_samples_split,
            trees: Vec::new(),
        }
    }

    fn bootstrap_sample(
        x: &Array2<f64>,
        y: &Array2<f64>,
        rng: &mut impl rand::Rng,
    ) -> (Array2<f64>, Array2<f64>) {
        let n_samples = x.nrows();
        let indices: Vec<usize> = (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect();

        let x_boot = Array2::from_shape_fn((n_samples, x.ncols()), |(i, j)| x[[indices[i], j]]);
        let y_boot = Array2::from_shape_fn((n_samples, y.ncols()), |(i, j)| y[[indices[i], j]]);

        (x_boot, y_boot)
    }
}

#[async_trait]
impl Model for RandomForestRegressor {
    async fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> MLResult<()> {
        if x_train.nrows() != y_train.nrows() {
            return Err(MLError::DimensionMismatch {
                expected: x_train.nrows(),
                actual: y_train.nrows(),
            });
        }

        self.trees.clear();
        let mut rng = rand::thread_rng();

        for _ in 0..self.n_trees {
            let (x_boot, y_boot) = Self::bootstrap_sample(x_train, y_train, &mut rng);

            let mut tree = DecisionTree::new(self.max_depth, self.min_samples_split);
            tree.fit(&x_boot, &y_boot)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    async fn predict(&self, x: &Array2<f64>) -> MLResult<Array2<f64>> {
        if self.trees.is_empty() {
            return Err(MLError::Prediction("模型未训练".to_string()));
        }

        let n_samples = x.nrows();
        let n_outputs = self.trees[0].n_outputs;
        let mut predictions = Array2::<f64>::zeros((n_samples, n_outputs));

        // 对每棵树的预测求平均
        for tree in &self.trees {
            let tree_pred = tree.predict(x)?;
            predictions = predictions + tree_pred;
        }

        predictions = predictions / self.n_trees as f64;

        Ok(predictions)
    }

    async fn save(&self, path: &str) -> MLResult<()> {
        let serialized = bincode::serialize(self)
            .map_err(|e| MLError::Serialization(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    async fn load(path: &str) -> MLResult<Self> {
        let data = std::fs::read(path)?;
        let model = bincode::deserialize(&data)
            .map_err(|e| MLError::Serialization(e.to_string()))?;
        Ok(model)
    }
}

/// 决策树节点
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TreeNode {
    Leaf {
        value: Array1<f64>,
    },
    Internal {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// 决策树回归模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    max_depth: usize,
    min_samples_split: usize,
    root: Option<TreeNode>,
    n_outputs: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            root: None,
            n_outputs: 0,
        }
    }

    fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> MLResult<()> {
        self.n_outputs = y.ncols();
        self.root = Some(self.build_tree(x, y, 0)?);
        Ok(())
    }

    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        depth: usize,
    ) -> MLResult<TreeNode> {
        let n_samples = x.nrows();

        // 停止条件
        if depth >= self.max_depth || n_samples < self.min_samples_split {
            let mean = y.mean_axis(ndarray::Axis(0)).ok_or_else(|| {
                MLError::Training("无法计算均值".to_string())
            })?;
            return Ok(TreeNode::Leaf { value: mean });
        }

        // 寻找最佳分割
        let (best_feature, best_threshold) = self.find_best_split(x, y)?;

        // 如果找不到好的分割，创建叶子节点
        if best_feature.is_none() {
            let mean = y.mean_axis(ndarray::Axis(0)).ok_or_else(|| {
                MLError::Training("无法计算均值".to_string())
            })?;
            return Ok(TreeNode::Leaf { value: mean });
        }

        let feature_idx = best_feature.unwrap();
        let threshold = best_threshold.unwrap();

        // 分割数据
        let (left_indices, right_indices) = self.split_data(x, feature_idx, threshold);

        if left_indices.is_empty() || right_indices.is_empty() {
            let mean = y.mean_axis(ndarray::Axis(0)).ok_or_else(|| {
                MLError::Training("无法计算均值".to_string())
            })?;
            return Ok(TreeNode::Leaf { value: mean });
        }

        let x_left = x.select(ndarray::Axis(0), &left_indices);
        let y_left = y.select(ndarray::Axis(0), &left_indices);
        let x_right = x.select(ndarray::Axis(0), &right_indices);
        let y_right = y.select(ndarray::Axis(0), &right_indices);

        let left = Box::new(self.build_tree(&x_left, &y_left, depth + 1)?);
        let right = Box::new(self.build_tree(&x_right, &y_right, depth + 1)?);

        Ok(TreeNode::Internal {
            feature_idx,
            threshold,
            left,
            right,
        })
    }

    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> MLResult<(Option<usize>, Option<f64>)> {
        let n_features = x.ncols();
        let mut best_mse = f64::INFINITY;
        let mut best_feature = None;
        let mut best_threshold = None;

        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);
            let unique_values: Vec<f64> = {
                let mut vals: Vec<f64> = feature_values.to_vec();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                vals.dedup();
                vals
            };

            for &threshold in &unique_values {
                let (left_indices, right_indices) = self.split_data(x, feature_idx, threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let y_left = y.select(ndarray::Axis(0), &left_indices);
                let y_right = y.select(ndarray::Axis(0), &right_indices);

                let mse = self.calculate_mse(&y_left) * left_indices.len() as f64
                    + self.calculate_mse(&y_right) * right_indices.len() as f64;

                if mse < best_mse {
                    best_mse = mse;
                    best_feature = Some(feature_idx);
                    best_threshold = Some(threshold);
                }
            }
        }

        Ok((best_feature, best_threshold))
    }

    fn split_data(&self, x: &Array2<f64>, feature_idx: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (i, row) in x.axis_iter(ndarray::Axis(0)).enumerate() {
            if row[feature_idx] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        (left_indices, right_indices)
    }

    fn calculate_mse(&self, y: &Array2<f64>) -> f64 {
        if y.is_empty() {
            return 0.0;
        }

        let mean = y.mean_axis(ndarray::Axis(0)).unwrap();
        let mut mse = 0.0;

        for row in y.axis_iter(ndarray::Axis(0)) {
            for (val, &mean_val) in row.iter().zip(mean.iter()) {
                mse += (val - mean_val).powi(2);
            }
        }

        mse / y.nrows() as f64
    }

    fn predict(&self, x: &Array2<f64>) -> MLResult<Array2<f64>> {
        if self.root.is_none() {
            return Err(MLError::Prediction("模型未训练".to_string()));
        }

        let n_samples = x.nrows();
        let mut predictions = Array2::<f64>::zeros((n_samples, self.n_outputs));

        for (i, row) in x.axis_iter(ndarray::Axis(0)).enumerate() {
            let pred = self.predict_single(&row, self.root.as_ref().unwrap())?;
            for (j, &val) in pred.iter().enumerate() {
                predictions[[i, j]] = val;
            }
        }

        Ok(predictions)
    }

    fn predict_single(&self, x: &ArrayView1<f64>, node: &TreeNode) -> MLResult<Array1<f64>> {
        match node {
            TreeNode::Leaf { value } => Ok(value.clone()),
            TreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if x[*feature_idx] <= *threshold {
                    self.predict_single(x, left)
                } else {
                    self.predict_single(x, right)
                }
            }
        }
    }
}

/// 线性回归模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    weights: Option<Array2<f64>>,
    bias: Option<Array1<f64>>,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: None,
        }
    }
}

#[async_trait]
impl Model for LinearRegression {
    async fn train(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>) -> MLResult<()> {
        use ndarray_linalg::LeastSquaresSvd;

        if x_train.nrows() != y_train.nrows() {
            return Err(MLError::DimensionMismatch {
                expected: x_train.nrows(),
                actual: y_train.nrows(),
            });
        }

        // 添加截距项
        let n_samples = x_train.nrows();
        let n_features = x_train.ncols();
        let mut x_with_bias = Array2::<f64>::ones((n_samples, n_features + 1));
        x_with_bias.slice_mut(ndarray::s![.., 1..]).assign(x_train);

        // 使用 SVD 求解最小二乘
        let solution = x_with_bias.least_squares(y_train)
            .map_err(|e| MLError::Training(format!("最小二乘求解失败: {:?}", e)))?;

        let params = solution.solution;
        self.bias = Some(params.row(0).to_owned());
        self.weights = Some(params.slice(ndarray::s![1.., ..]).to_owned());

        Ok(())
    }

    async fn predict(&self, x: &Array2<f64>) -> MLResult<Array2<f64>> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| MLError::Prediction("模型未训练".to_string()))?;
        let bias = self.bias.as_ref()
            .ok_or_else(|| MLError::Prediction("模型未训练".to_string()))?;

        let predictions = x.dot(weights) + bias;
        Ok(predictions)
    }

    async fn save(&self, path: &str) -> MLResult<()> {
        let serialized = bincode::serialize(self)
            .map_err(|e| MLError::Serialization(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    async fn load(path: &str) -> MLResult<Self> {
        let data = std::fs::read(path)?;
        let model = bincode::deserialize(&data)
            .map_err(|e| MLError::Serialization(e.to_string()))?;
        Ok(model)
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}
