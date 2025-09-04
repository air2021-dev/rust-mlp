# rust_mlp

[![CI](https://github.com/air2021-dev/rust-mlp/actions/workflows/ci.yml/badge.svg)](https://github.com/air2021-dev/rust-mlp/actions/workflows/ci.yml)

A minimal, from-scratch MLP in Rust:
- Safe Tensor with 2D ops (matmul, transpose, softmax, etc.)
- Layers with weights/biases and activations (ReLU, Sigmoid, Softmax)
- Forward, Backward (with Softmax + Cross-Entropy simplification), and SGD training
- Example: XOR

## Build and run

```bash
cargo run --example xor
```

You should see the loss decreasing and final predictions approaching the correct XOR mapping.

## API sketch

```rust
use rust_mlp::{Activation, Network, Tensor};

let x = Tensor::from_vec(batch, in_dim, data);
let y = Tensor::from_vec(batch, out_dim, labels_one_hot);

let sizes = [in_dim, 64, out_dim];
let activations = [Activation::ReLU, Activation::Softmax];
let mut net = Network::new(&sizes, &activations);

net.train(&x, &y, 1000, 0.01, true);
let (pred, _cache) = net.forward(&x);
```

## Notes

- For performance, this uses simple safe Rust loops. You can later add:
  - rayon for parallel matmul
  - BLAS via cblas/accelerate (unsafe FFI) behind a feature flag
- This is intentionally minimal to be easy to extend (optimizers, regularization, mini-batching, etc.).

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License in the LICENSE file or at http://www.apache.org/licenses/LICENSE-2.0