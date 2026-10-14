//! Dependency-free inference for the congestion CNN exported from PyTorch (stack of dilated
//! "same"-padded 2-D convolutions). Used natively for benchmarks; the browser runs the same
//! network through ONNX Runtime Web, and CI checks both agree.

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct ConvLayer {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub dilation: usize,
    /// row-major [cout][cin][k][k]
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    /// "relu" | "sigmoid" | "none"
    pub act: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Cnn {
    pub layers: Vec<ConvLayer>,
}

impl Cnn {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// x: CHW tensor. Returns a single-channel HxW map.
    pub fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        for l in &self.layers {
            cur = conv2d(&cur, l, h, w);
        }
        cur
    }
}

fn conv2d(x: &[f32], l: &ConvLayer, h: usize, w: usize) -> Vec<f32> {
    let hw = h * w;
    let k = l.k as isize;
    let pad = (l.dilation as isize) * (k / 2);
    let mut y = vec![0.0f32; l.cout * hw];
    for o in 0..l.cout {
        let out = &mut y[o * hw..(o + 1) * hw];
        out.iter_mut().for_each(|v| *v = l.bias[o]);
        for i in 0..l.cin {
            let inp = &x[i * hw..(i + 1) * hw];
            for ky in 0..k {
                for kx in 0..k {
                    let wv = l.weight[((o * l.cin + i) * l.k + ky as usize) * l.k + kx as usize];
                    if wv == 0.0 {
                        continue;
                    }
                    let dy = ky * l.dilation as isize - pad;
                    let dx = kx * l.dilation as isize - pad;
                    for yy in 0..h as isize {
                        let sy = yy + dy;
                        if sy < 0 || sy >= h as isize {
                            continue;
                        }
                        let row_o = yy as usize * w;
                        let row_i = sy as usize * w;
                        for xx in 0..w as isize {
                            let sx = xx + dx;
                            if sx >= 0 && sx < w as isize {
                                out[row_o + xx as usize] += wv * inp[row_i + sx as usize];
                            }
                        }
                    }
                }
            }
        }
        match l.act.as_str() {
            "relu" => out.iter_mut().for_each(|v| *v = v.max(0.0)),
            "sigmoid" => out.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp())),
            _ => {}
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_kernel_with_padding() {
        // 3x3 kernel with a single 1 at centre must copy the input
        let mut wt = vec![0.0; 9];
        wt[4] = 1.0;
        let l = ConvLayer {
            cin: 1,
            cout: 1,
            k: 3,
            dilation: 2,
            weight: wt,
            bias: vec![0.0],
            act: "none".into(),
        };
        let x: Vec<f32> = (0..12).map(|v| v as f32).collect();
        assert_eq!(conv2d(&x, &l, 3, 4), x);
    }

    #[test]
    fn shift_kernel_zero_pads() {
        // weight at (ky=1,kx=2): output(y,x) = input(y, x+1)
        let mut wt = vec![0.0; 9];
        wt[5] = 1.0;
        let l = ConvLayer {
            cin: 1,
            cout: 1,
            k: 3,
            dilation: 1,
            weight: wt,
            bias: vec![0.5],
            act: "none".into(),
        };
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(conv2d(&x, &l, 1, 3), vec![2.5, 3.5, 0.5]);
    }
}
