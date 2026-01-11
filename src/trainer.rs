use crate::model::Model;
use crate::optimizers::base_optimizer::AbstractOptimizerTrait;
use crate::losses::base_loss::AbstractLossFunctionTrait;
use crate::data::DataSet;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Trainer<O, L>
where
    O: AbstractOptimizerTrait,
    L: AbstractLossFunctionTrait,
{
    pub model: Model,
    pub optimizer: O,
    pub loss_function: L,
    pub train_dataset: DataSet,
    pub test_dataset: DataSet,
    // pub device: Device,
    pub epoch: usize,
    pub batch_size: usize,
    pub verbose: bool,
    pub eval_limit: Option<usize>,
    pub train_limit: Option<usize>,
    // pub callbacks: Vec<Box<dyn Callback>>,
    // pub metrics: Vec<Box<dyn Metric>>,
    // pub visualization: bool,
}

impl<O, L> Trainer<O, L> where
    O: AbstractOptimizerTrait,
    L: AbstractLossFunctionTrait,
{
    pub fn new(
        model: Model, 
        optimizer: O, 
        loss_function: L, 
        train_dataset: DataSet, 
        test_dataset: DataSet, 
        epoch: usize, 
        batch_size: usize,
        verbose: bool,
        debug: bool
    ) -> Self {
        let eval_limit = if debug { Some(1000) } else { None };
        let train_limit = if debug { Some(100) } else { None };
        Self {
            model,
            optimizer,
            loss_function,
            train_dataset,
            test_dataset,
            epoch,
            batch_size,
            verbose,
            eval_limit,
            train_limit,
        }
    }

    pub fn run(&mut self) {
        
        let output_size = self.model.layers.last().unwrap().o_size();
        let num_features = self.train_dataset.num_features;

        let train_num_samples = self.train_limit
            .unwrap_or(self.train_dataset.num_samples)
            .min(self.train_dataset.num_samples);

        let batch_size = self.batch_size.max(1);
        let num_batches = (train_num_samples + batch_size - 1) / batch_size;

        for epoch in 0..self.epoch {
            if self.verbose {
                println!("\n{}", "=".repeat(60));
                println!("Epoch {}/{}", epoch + 1, self.epoch);
                println!("{}", "=".repeat(60));
            }

            let mut indices: Vec<usize> = (0..train_num_samples).collect();
            indices.shuffle(&mut thread_rng());

            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(train_num_samples);
                let current_batch_size = end - start;

                let mut batch_grad_w: Vec<Vec<f32>> = self.model.layers
                    .iter()
                    .map(|layer| vec![0.0; layer.w().len()])
                    .collect();
                let mut batch_grad_b: Vec<Vec<f32>> = self.model.layers
                    .iter()
                    .map(|layer| vec![0.0; layer.b().len()])
                    .collect();
                let mut loss_sum: f32 = 0.0;
                let mut last_output: Vec<f32> = Vec::new();
                let mut last_label: Vec<f32> = Vec::new();

                for i in start..end {
                    let sample_idx = indices[i];
                    // prepare input and label
                    let input_start = sample_idx * num_features;
                    let input_end = input_start + num_features;
                    let input = self.train_dataset.images[input_start..input_end].to_vec();
                    let mut label = vec![0.0; output_size];
                    label[self.train_dataset.labels[sample_idx] as usize] = 1.0;

                    // forward
                    let output = self.model.forward(&input);

                    // calculate loss
                    let loss = self.loss_function.forward(&label, &output);
                    loss_sum += loss;
                    last_output = output.clone();
                    last_label = label.clone();

                    // backward
                    let loss_grad = self.loss_function.backward(&label, &output);
                    self.model.backward(&loss_grad);

                    // accumulate gradients
                    for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                        for j in 0..layer.grad_w().len() {
                            batch_grad_w[layer_idx][j] += layer.grad_w()[j];
                        }
                        for j in 0..layer.grad_b().len() {
                            batch_grad_b[layer_idx][j] += layer.grad_b()[j];
                        }
                    }
                }

                let scale = 1.0 / current_batch_size as f32;
                for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
                    {
                        let grad_w = layer.grad_w_mut();
                        for j in 0..grad_w.len() {
                            grad_w[j] = batch_grad_w[layer_idx][j] * scale;
                        }
                    }
                    {
                        let grad_b = layer.grad_b_mut();
                        for j in 0..grad_b.len() {
                            grad_b[j] = batch_grad_b[layer_idx][j] * scale;
                        }
                    }
                }

                // update
                self.optimizer.update(&mut self.model);

                // verbose output
                if self.verbose && batch_idx % 10 == 0 {
                    let avg_loss = loss_sum / current_batch_size as f32;
                    self.verbose_output(epoch, batch_idx, num_batches, avg_loss, &last_output, &last_label);
                }
            }

            let accuracy = self.evaluate_accuracy();
            if self.verbose {
                println!("Validation accuracy: {:.2}%", accuracy);
            }

            if self.verbose {
                println!("\n{}", "-".repeat(60));
            }
        }
    }

    fn verbose_output(&self, _epoch: usize, i: usize, num_steps: usize, loss: f32, output: &[f32], label: &[f32]) {
        let progress = (i as f32 / num_steps as f32) * 100.0;
        println!("\n[Step {}/{} ({:.1}%)]", i + 1, num_steps, progress);
        println!("  Loss: {:.6}", loss);
        
        // Find predicted class
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let true_class = label.iter()
            .position(|&x| x == 1.0)
            .unwrap();
        
        println!("  True class: {}, Predicted class: {}", true_class, predicted_class);
        println!("  Output probabilities: {:?}", 
            output.iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
    }

    fn evaluate_accuracy(&mut self) -> f32 {
        let mut correct = 0usize;
        let eval_samples = self.eval_limit.unwrap_or(self.test_dataset.num_samples).min(self.test_dataset.num_samples);
        for idx in 0..eval_samples {
            let input_start = idx * self.test_dataset.num_features;
            let input_end = input_start + self.test_dataset.num_features;
            let input = self.test_dataset.images[input_start..input_end].to_vec();
            let label = self.test_dataset.labels[idx];
            let output = self.model.forward(&input);
            let predicted_class = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if predicted_class == label as usize {
                correct += 1;
            }
        }
        let denom = if eval_samples == 0 { 1 } else { eval_samples };
        (correct as f32 / denom as f32) * 100.0
    }
}
