import tensorflow as tf
from keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention
from keras.models import Model
from feature_extractor import FeatureExtractor
from fusion import Fusion

class Transformer(Model):
    def __init__(self, num_classes=2, num_frames=8, embed_dims=512, num_heads=8,
                  ff_dim=2048, num_transformer_layers=4, dropout_rate=0.1, use_spatial_attention=True):
        super(Transformer, self).__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_layers = num_transformer_layers
        self.use_spatial_attention = use_spatial_attention

        self.temporal_transformer_layers = []
        for _ in range(num_transformer_layers):
            self.temporal_transformer_layers.append([
                MultiHeadAttention(num_heads=num_heads, key_dim=embed_dims // num_heads),
                LayerNormalization(epsilon=1e-6),
                Dense(ff_dim, activation='relu', kernel_initializer='he_normal'),
                Dropout(dropout_rate),
                Dense(embed_dims, kernel_initializer='he_normal'),
                Dropout(dropout_rate),
                LayerNormalization(epsilon=1e-6)
            ])
        
        if use_spatial_attention:
            self.spatial_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dims // num_heads)
            self.spatial_norm = LayerNormalization(epsilon=1e-6)

        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')

    def call(self, inputs, training=False):
        # inputs shape: [batch_size, num_frames, embed_dims]
        x = inputs

        if self.use_spatial_attention:
            x = tf.transpose(x, [0, 2, 1])  # [batch_size, embed_dims, num_frames]
            x = self.spatial_attention(x, x, training=training)
            x = self.spatial_norm(x+x, training=training)
            x = tf.transpose(x, [0, 2, 1])

        for layer in self.temporal_transformer_layers:
            attn_output = layer[0](x, x, training=training)
            x = layer[1](x + attn_output, training=training)
            ff_output = layer[2](x, training=training)
            ff_output = layer[3](ff_output, training=training)
            ff_output = layer[4](ff_output, training=training)
            x = layer[5](x + ff_output, training=training)

        x = self.global_pool(x)  # [batch_size, embed_dims]
        x = self.classifier(x)  # [batch_size, num_classes]

        return x
    
def create_transformer_model(num_classes=2, num_frames=8, embed_dims=512, num_heads=8,
                             ff_dim=2048, num_transformer_layers=4, dropout_rate=0.1, use_spatial_attention=True):
    
    inputs = tf.keras.Input(shape=(num_frames, 299 , 299, 3))

    feature_extractor = FeatureExtractor()
    xception_features, efficientnet_features = feature_extractor(inputs)

    fusion = Fusion(embed_dims=embed_dims)
    fusion_features = fusion(xception_features, efficientnet_features)

    transformer = Transformer(num_classes=num_classes, num_frames=num_frames, embed_dims=embed_dims,
                             num_heads=num_heads, ff_dim=ff_dim, num_transformer_layers=num_transformer_layers,
                                dropout_rate=dropout_rate, use_spatial_attention=use_spatial_attention)
    outputs = transformer(fusion_features)
    return Model(inputs=inputs, outputs=outputs, name='TransformerModel')

if __name__ == "__main__":
    model = create_transformer_model(
        num_classes=2,
        num_frames=8,
        embed_dims=512,
        num_heads=8,
        ff_dim=2048,
        num_transformer_layers=4,
        dropout_rate=0.1,
        use_spatial_attention=True)
    model.summary()
            



