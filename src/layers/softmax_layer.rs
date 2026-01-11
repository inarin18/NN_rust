use crate::layers::base_layer::{AbstractLayer, AbstractLayerTrait};
use crate::activation::softmax;

#[derive(Debug)]
pub struct SoftmaxLayer {
    base: AbstractLayer,
}

impl SoftmaxLayer {
    pub fn new(name: String, i_size: usize, o_size: usize) -> Self {
        Self {
            base: AbstractLayer::new(name, i_size, o_size, "softmax".to_string()),
        }
    }
}

impl AbstractLayerTrait for SoftmaxLayer {
    fn build(&mut self) {}

    fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        // 入力を保存
        self.base.last_input = x.to_vec();
        
        // softmax を計算
        let output = softmax(x);
        
        // 出力を保存（backward で使用）
        self.base.last_output = output.clone();
        output
    }

    fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        // softmax の出力（forward で保存したもの）
        let s = &self.base.last_output;
        
        // softmax の勾配: ∂s_i / ∂x_j = s_i * (δ_ij - s_j)
        // チェインルール: ∂L / ∂x_j = Σ_i (∂L / ∂s_i) * (∂s_i / ∂x_j)
        //                = s_j * (grad_output[j] - Σ_i grad_output[i] * s_i)
        
        // Σ_i grad_output[i] * s_i を計算
        let sum_grad_s: f32 = grad_output.iter()
            .zip(s.iter())
            .map(|(g, s_i)| g * s_i)
            .sum();
        
        // 各入力に対する勾配を計算
        let grad_input: Vec<f32> = s.iter()
            .zip(grad_output.iter())
            .map(|(s_j, grad_j)| {
                s_j * (grad_j - sum_grad_s)
            })
            .collect();
        
        grad_input
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn i_size(&self) -> usize {
        self.base.i_size
    }

    fn o_size(&self) -> usize {
        self.base.o_size
    }

    fn w(&self) -> &Vec<f32> {
        &self.base.w
    }

    fn b(&self) -> &Vec<f32> {
        &self.base.b
    }

    fn grad_w(&self) -> &Vec<f32> {
        &self.base.grad_w
    }

    fn grad_b(&self) -> &Vec<f32> {
        &self.base.grad_b
    }

    fn grad_w_mut(&mut self) -> &mut Vec<f32> {
        &mut self.base.grad_w
    }

    fn grad_b_mut(&mut self) -> &mut Vec<f32> {
        &mut self.base.grad_b
    }

    fn update_weights(&mut self, delta_w: &[f32]) {
        for i in 0..self.base.w.len() {
            self.base.w[i] += delta_w[i];
        }
    }

    fn update_biases(&mut self, delta_b: &[f32]) {
        for i in 0..self.base.b.len() {
            self.base.b[i] += delta_b[i];
        }
    }

    fn activation_type(&self) -> &str {
        &self.base.activation_type
    }
}
