use rust_mlp::{Activation, Network, Tensor};

#[test]
fn softmax_rows_sum_to_one() {
    let x = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
    let s = x.softmax_rows();
    for r in 0..s.rows {
        let mut sum = 0.0f32;
        for c in 0..s.cols {
            sum += s.get(r, c);
        }
        assert!((sum - 1.0).abs() < 1e-5, "row {} sum {}", r, sum);
    }
}

#[test]
fn network_forward_smoke() {
    let x = Tensor::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let sizes = [2, 4, 2];
    let activations = [Activation::ReLU, Activation::Softmax];
    let net = Network::new(&sizes, &activations);
    let (pred, _cache) = net.forward(&x);
    assert_eq!(pred.rows, 4);
    assert_eq!(pred.cols, 2);
    for v in pred.data {
        assert!(v.is_finite());
        assert!((0.0 - 1e-6..=1.0 + 1e-6).contains(&v));
    }
}
