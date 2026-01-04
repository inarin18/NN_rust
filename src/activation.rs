pub fn identity(x: f32) -> f32 { x }

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    // 数値安定性のため、最大値を引く
    let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    let exp_x: Vec<f32> = x.iter()
        .map(|&xi| (xi - max_x).exp())
        .collect();
    
    let sum_exp_x: f32 = exp_x.iter().sum();
    
    exp_x.iter()
        .map(|&exp_val| exp_val / sum_exp_x)
        .collect()
}