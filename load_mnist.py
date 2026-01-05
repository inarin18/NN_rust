# load_mnist.py
from sklearn.datasets import fetch_openml
import numpy as np
import struct

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

# データを正規化
X_normalized: np.ndarray = (X.astype(np.float32) / 255.0)
y_int: np.ndarray = y.astype(np.uint8)

# シンプルなバイナリ形式で保存
with open('data/mnist.bin', 'wb') as f:
    # ヘッダー: [num_samples (u64), num_features (u64)]
    f.write(struct.pack('QQ', X.shape[0], X.shape[1]))
    # 画像データ: [num_samples * num_features] f32
    X_normalized.flatten().tofile(f)
    # ラベル: [num_samples] u8
    y_int.tofile(f)

print(f"Saved {X.shape[0]} samples to data/mnist.bin")