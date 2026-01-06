// src/data.rs
use std::fs::File;
use std::io::{BufReader, Read};

pub struct DataSet {
    pub images: Vec<f32>,  // フラットな配列
    pub labels: Vec<u8>,
    pub num_samples: usize,
    pub num_features: usize,
}

impl DataSet {

    pub fn new(images: Vec<f32>, labels: Vec<u8>, num_features: usize) -> Self {
        let num_samples = images.len() / num_features as usize;
        Self { images, labels, num_samples, num_features }
    }

    pub fn load_from_binary(filename: &str) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(filename)?);
        
        // ヘッダー読み込み: [num_samples (u64), num_features (u64)]
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;
        
        let num_samples = u64::from_le_bytes([
            header[0], header[1], header[2], header[3],
            header[4], header[5], header[6], header[7]
        ]) as usize;
        
        let num_features = u64::from_le_bytes([
            header[8], header[9], header[10], header[11],
            header[12], header[13], header[14], header[15]
        ]) as usize;
        
        // 画像データ読み込み (f32 = 4 bytes)
        let image_size = num_samples * num_features * 4;
        let mut image_bytes = vec![0u8; image_size];
        file.read_exact(&mut image_bytes)?;
        
        let images: Vec<f32> = image_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // ラベル読み込み
        let mut labels = vec![0u8; num_samples];
        file.read_exact(&mut labels)?;
        
        Ok(DataSet {
            images,
            labels,
            num_samples,
            num_features,
        })
    }
    
    pub fn get_image(&self, index: usize) -> Option<&[f32]> {
        if index < self.num_samples {
            let start = index * self.num_features;
            let end = start + self.num_features;
            Some(&self.images[start..end])
        } else {
            None
        }
    }
    
    pub fn get_label(&self, index: usize) -> Option<u8> {
        self.labels.get(index).copied()
    }

    pub fn split_dataset(&self, ratio: f32) -> (DataSet, DataSet) {
        let num_samples = self.num_samples;
        let num_train_samples = (num_samples as f32 * ratio) as usize;
        let (train_images, test_images) = self.images.split_at(num_train_samples * self.num_features);
        let (train_labels, test_labels) = self.labels.split_at(num_train_samples);
        (
            DataSet::new(train_images.to_vec(), train_labels.to_vec(), self.num_features), 
            DataSet::new(test_images.to_vec(), test_labels.to_vec(), self.num_features)
        )
    }
    
    /// 画像データをアスキーアートとして表示する
    /// 
    /// # Arguments
    /// * `image` - 28x28の画像データ（784要素の配列）
    /// * `label` - オプションのラベル（表示される数字）
    pub fn display_ascii_art(image: &[f32], label: Option<u8>) {
        const WIDTH: usize = 28;
        const HEIGHT: usize = 28;
        
        // ラベルを表示
        if let Some(l) = label {
            println!("\nLabel: {}", l);
        }
        println!("┌{}┐", "─".repeat(WIDTH));
        
        // 画像を表示
        for y in 0..HEIGHT {
            print!("│");
            for x in 0..WIDTH {
                let idx = y * WIDTH + x;
                let pixel = if idx < image.len() { image[idx] } else { 0.0 };
                
                // ピクセル値を0.0-1.0の範囲で文字に変換
                let char = if pixel < 0.1 {
                    " "
                } else if pixel < 0.3 {
                    "."
                } else if pixel < 0.5 {
                    ":"
                } else if pixel < 0.7 {
                    "+"
                } else if pixel < 0.9 {
                    "*"
                } else {
                    "#"
                };
                print!("{}", char);
            }
            println!("│");
        }
        
        println!("└{}┘", "─".repeat(WIDTH));
    }
    
    /// インデックスを指定して画像をアスキーアートとして表示する
    pub fn display_image(&self, index: usize) {
        if let Some(image) = self.get_image(index) {
            let label = self.get_label(index);
            Self::display_ascii_art(image, label);
        } else {
            println!("Invalid index: {}", index);
        }
    }
}