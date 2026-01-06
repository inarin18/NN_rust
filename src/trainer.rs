use crate::model::Model;
use crate::optimizers::base_optimizer::AbstractOptimizerTrait;
use crate::losses::base_loss::AbstractLossFunctionTrait;
use crate::data::DataSet;

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
    // pub batch_size: usize,
    pub verbose: bool,
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
        verbose: bool
    ) -> Self {
        Self { model, optimizer, loss_function, train_dataset, test_dataset, epoch, verbose }
    }

    pub fn run(&mut self) {
        
        let output_size = self.model.layers.last().unwrap().o_size();
        let num_features = self.train_dataset.num_features;

        let train_num_samples = self.train_dataset.num_samples;
        let train_images = &self.train_dataset.images.chunks(num_features).collect::<Vec<&[f32]>>();
        let train_labels = &self.train_dataset.labels;

        // let test_images = &self.test_dataset.images;
        // let test_labels = &self.test_dataset.labels;

        for epoch in 0..self.epoch {
            if self.verbose {
                println!("\n{}", "=".repeat(60));
                println!("Epoch {}/{}", epoch + 1, self.epoch);
                println!("{}", "=".repeat(60));
            }

            for i in 0..train_num_samples {

                // prepare input and label
                let input = train_images[i].to_vec();
                let mut label = vec![0.0; output_size];
                label[train_labels[i] as usize] = 1.0;

                // forward
                let output = self.model.forward(&input);

                // calculate loss
                let loss = self.loss_function.forward(&label, &output);

                // verbose output
                if self.verbose && i % 100 == 0 {
                    self.verbose_output(epoch, i, train_num_samples, loss, &output, &label);
                }

                // backward
                let loss_grad = self.loss_function.backward(&label, &output);
                self.model.backward(&loss_grad);

                // update
                self.optimizer.update(&mut self.model);
            }

            if self.verbose {
                println!("\n{}", "-".repeat(60));
            }
        }
    }

    fn verbose_output(&self, epoch: usize, i: usize, train_num_samples: usize, loss: f32, output: &[f32], label: &[f32]) {
        let progress = (i as f32 / train_num_samples as f32) * 100.0;
        println!("\n[Step {}/{} ({:.1}%)]", i + 1, train_num_samples, progress);
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
}