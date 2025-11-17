"""
Lightweight ST-GCN for skeleton-based Sign Language Recognition.
Input shape: [batch, T, V, C]
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Graph Convolution Layer ---
class GraphConv(layers.Layer):
    def __init__(self, A, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.A = tf.constant(A, dtype=tf.float32)  # adjacency (K, V, V)
        self.out_channels = out_channels
        self.conv = layers.Conv2D(out_channels, (1, 1))

    def call(self, x):
        K = self.A.shape[0]
        out = 0
        for k in range(K):
            x_k = tf.einsum('ntvc,vw->ntwc', x, self.A[k])
            x_k = self.conv(x_k)
            out += x_k
        return out

    def get_config(self):
        config = super().get_config()
        # Serialize adjacency as a list
        config.update({
            "A": self.A.numpy().tolist(),
            "out_channels": self.out_channels
        })
        return config


# --- ST-GCN block ---
def st_gcn_block(x, A, out_channels, stride=1):
    # Spatial GCN
    x_spatial = GraphConv(A, out_channels)(x)
    x_spatial = layers.BatchNormalization()(x_spatial)
    x_spatial = layers.ReLU()(x_spatial)

    # Temporal Convolution
    x_temporal = layers.Conv2D(out_channels, (9, 1), padding='same', strides=(stride,1))(x_spatial)
    x_temporal = layers.BatchNormalization()(x_temporal)

    # Residual connection
    if x.shape[-1] != out_channels or stride != 1:
        res = layers.Conv2D(out_channels, (1,1), strides=(stride,1))(x)
        res = layers.BatchNormalization()(res)
    else:
        res = x

    x_out = layers.ReLU()(x_temporal + res)
    return x_out

# --- Build ST-GCN Model ---
def build_stgcn(T=75, V=75, C=3, num_classes=142):
    # 3 adjacency partitions: identity, small weights, zeros
    A = tf.stack([
        tf.eye(V),              # self-connections
        tf.ones((V, V)) * 0.01, # placeholder adjacency
        tf.zeros((V, V))
    ])

    inp = layers.Input(shape=(T, V, C))  # (batch, T, V, C)
    
    x = inp
    x = st_gcn_block(x, A, 64)
    x = st_gcn_block(x, A, 128, stride=2)
    x = st_gcn_block(x, A, 256, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)
    return model
