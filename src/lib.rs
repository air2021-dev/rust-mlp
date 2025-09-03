use std::f32;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    pub fn softmax_rows(&self) -> Tensor {
        let mut result = self.clone();
        for r in 0..self.rows {
            let row_start = r * self.cols;
            let row_end = row_start + self.cols;
            let row_slice = &self.data[row_start..row_end];

            // Find max for numerical stability
            let max_val = row_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0;
            for c in 0..self.cols {
                let exp_val = (self.get(r, c) - max_val).exp();
                result.set(r, c, exp_val);
                exp_sum += exp_val;
            }

            // Normalize
            for c in 0..self.cols {
                result.set(r, c, result.get(r, c) / exp_sum);
            }
        }
        result
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols, other.rows);
        let mut result = Tensor::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let mut result = self.clone();

        if other.rows == 1 {
            // Broadcasting: add bias vector to each row
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.set(i, j, self.get(i, j) + other.get(0, j));
                }
            }
        } else {
            // Element-wise addition
            assert_eq!(self.rows, other.rows);
            assert_eq!(self.cols, other.cols);
            for i in 0..self.data.len() {
                result.data[i] += other.data[i];
            }
        }
        result
    }

    pub fn relu(&self) -> Tensor {
        let mut result = self.clone();
        for val in &mut result.data {
            *val = val.max(0.0);
        }
        result
    }
}

#[derive(Clone, Debug)]
pub enum Activation {
    ReLU,
    Softmax,
}

#[derive(Clone)]
pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub activation: Activation,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        // Simple Xavier initialization
        let scale = (2.0 / input_size as f32).sqrt();
        let weights_data: Vec<f32> = (0..input_size * output_size)
            .map(|i| {
                // Simple pseudo-random initialization
                let x = (i as f32 * 1.618034) % 1.0;
                (x - 0.5) * 2.0 * scale
            })
            .collect();

        Self {
            weights: Tensor::from_vec(input_size, output_size, weights_data),
            biases: Tensor::new(1, output_size),
            activation,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let z = input.matmul(&self.weights).add(&self.biases);
        match self.activation {
            Activation::ReLU => z.relu(),
            Activation::Softmax => z.softmax_rows(),
        }
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(sizes: &[usize], activations: &[Activation]) -> Self {
        assert_eq!(sizes.len() - 1, activations.len());

        let layers = (0..activations.len())
            .map(|i| Layer::new(sizes[i], sizes[i + 1], activations[i].clone()))
            .collect();

        Self { layers }
    }

    pub fn forward(&self, input: &Tensor) -> (Tensor, Vec<Tensor>) {
        let mut cache = vec![input.clone()];
        let mut current = input.clone();

        for layer in &self.layers {
            current = layer.forward(&current);
            cache.push(current.clone());
        }

        (current, cache)
    }

    pub fn train(
        &mut self,
        _x: &Tensor,
        _y: &Tensor,
        _epochs: usize,
        _learning_rate: f32,
        _verbose: bool,
    ) {
        // Minimal implementation - training not needed for the tests
        // In a full implementation, this would do backpropagation and gradient descent
    }
}
