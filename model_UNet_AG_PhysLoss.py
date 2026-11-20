import tensorflow as tf
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="custom")
class CharbMaskedSSIMGradLoss(tf.keras.losses.Loss):
    def __init__(self,
                 w_charb=0.65,
                 w_ssim=0.25,
                 w_grad=0.10,
                 eps=1e-3,
                 mask_k=8.0,
                 name="Charb_MaskedSSIM_Grad"):
        super().__init__(name=name)
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.eps = eps
        self.mask_k = mask_k

    # ---------- soft physical mask ----------
    def _co_mask(self, y_true):
        """
        物理意义：
        - 高于背景显著值的 CO 区域
        """
        mean = tf.reduce_mean(y_true, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(y_true, axis=[1, 2, 3], keepdims=True)

        threshold = mean + 0.5 * std
        mask = tf.sigmoid(self.mask_k * (y_true - threshold))
        return mask

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true = tf.where(tf.math.is_finite(y_true), y_true, 0.0)
        y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)

        # ---------- Charbonnier ----------
        diff = y_true - y_pred
        l_charb = tf.reduce_mean(
            tf.sqrt(diff * diff + self.eps * self.eps)
        )

        # ---------- Masked SSIM ----------
        mask = self._co_mask(y_true)

        y_true_m = y_true * mask
        y_pred_m = y_pred * mask

        ssim = tf.image.ssim(
            y_true_m,
            y_pred_m,
            max_val=1.0,
            filter_size=7
        )
        l_ssim = 1.0 - tf.reduce_mean(ssim)

        # ---------- Gradient loss ----------
        dy_t, dx_t = tf.image.image_gradients(y_true)
        dy_p, dx_p = tf.image.image_gradients(y_pred)

        grad_diff = (
            tf.abs(dx_t - dx_p) +
            tf.abs(dy_t - dy_p)
        )

        # 梯度 loss 也只在 mask 区域
        l_grad = tf.reduce_mean(grad_diff * mask)

        return (
            self.w_charb * l_charb +
            self.w_ssim * l_ssim +
            self.w_grad * l_grad
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "w_charb": self.w_charb,
            "w_ssim": self.w_ssim,
            "w_grad": self.w_grad,
            "eps": self.eps,
            "mask_k": self.mask_k
        })
        return cfg

# =====================================================
# 稳定版 Residual UNet + Attention Gate
# =====================================================
class CustomUNet:
    def __init__(self,
                 input_shape=(159, 219, 52),
                 norm_layer=None,
                 dropout_rate=0.0):
        self.input_shape = input_shape
        self.norm_layer = norm_layer
        self.dropout_rate = dropout_rate

    # ---------- Residual Block ----------
    def residual_block(self, x, filters):
        shortcut = x

        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

        if self.dropout_rate > 0:
            x = layers.SpatialDropout2D(self.dropout_rate)(x)

        return x

    # ---------- Safe Attention Gate ----------
    def attention_gate(self, g, x, inter_channels):
        """
        g: decoder feature
        x: encoder skip
        inter_channels: 固定中间通道数（关键）
        """
        theta_x = layers.Conv2D(
            inter_channels, 1, padding="same")(x)
        phi_g = layers.Conv2D(
            inter_channels, 1, padding="same")(g)

        f = layers.Add()([theta_x, phi_g])
        f = layers.ReLU()(f)

        psi = layers.Conv2D(1, 1, padding="same",
                            activation="sigmoid")(f)
        return layers.Multiply()([x, psi])

    # ---------- Build ----------
    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = self.norm_layer(inputs) if self.norm_layer else inputs

        # padding
        pad_h = (8 - self.input_shape[0] % 8) % 8
        pad_w = (8 - self.input_shape[1] % 8) % 8
        x = layers.ZeroPadding2D(((0, pad_h), (0, pad_w)))(x)

        # ===== Encoder =====
        c0 = self.residual_block(x, 64)
        p1 = layers.MaxPooling2D()(c0)

        c1 = self.residual_block(p1, 128)
        p2 = layers.MaxPooling2D()(c1)

        c2 = self.residual_block(p2, 256)
        p3 = layers.MaxPooling2D()(c2)

        bn = self.residual_block(p3, 384)

        # ===== Decoder + AG =====
        u1 = layers.UpSampling2D()(bn)
        c2_ag = self.attention_gate(u1, c2, inter_channels=128)
        u1 = layers.Concatenate()([u1, c2_ag])
        u1 = self.residual_block(u1, 256)

        u2 = layers.UpSampling2D()(u1)
        c1_ag = self.attention_gate(u2, c1, inter_channels=64)
        u2 = layers.Concatenate()([u2, c1_ag])
        u2 = self.residual_block(u2, 128)

        u3 = layers.UpSampling2D()(u2)
        c0_ag = self.attention_gate(u3, c0, inter_channels=32)
        u3 = layers.Concatenate()([u3, c0_ag])
        u3 = self.residual_block(u3, 64)

        out = layers.Conv2D(1, 1, activation="relu")(u3)
        out = layers.Cropping2D(((0, pad_h), (0, pad_w)))(out)

        return models.Model(inputs, out)