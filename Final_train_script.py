from extract_2train import *
from model_UNet-AG-PhysLoss import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
#import tensorflow_addons as tfa
import os

if __name__ == "__main__":
    try:
        path = "/share/home/fengsz1/zhuanghl/RAPASv2_ensrf/ensrfv2/working"
        save_path = "/share/home/fengsz1/zhuanghl/AGT_UNet"
        startdate = 2021122600
        enddate = 2024020600
        train_days = 114

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

        # 加载所有数据
        x_train, y_train, x_test, y_test = data_assemble(path, train_days, startdate, enddate)

        norm_layer = Normalization(axis=-1)
        norm_layer.adapt(x_train)

        np.save("/share/home/fengsz1/zhuanghl/AGT_UNet/contract_test/0_2/norm_mean.npy", norm_layer.mean.numpy())
        np.save("/share/home/fengsz1/zhuanghl/AGT_UNet/contract_test/0_2/norm_var.npy", norm_layer.variance.numpy())

        model = CustomUNet(
            input_shape=(159, 219, x_train.shape[-1]),
            norm_layer=norm_layer,
            dropout_rate=0.0
        ).build_model()

        model.compile(
                optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5, #5e-5
                clipnorm=1.0
            ),
            loss=CharbMaskedSSIMGradLoss(
                w_charb=0.7,
                w_ssim=0.2,
                w_grad=0.1,
                mask_k=4.0
            )
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                min_delta=1e-4,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=7,
                cooldown=2,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                filepath=f'{save_path}/contract_test/0_2/CO_model_{{epoch:03d}}_{{val_loss:.4f}}.keras',
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True,
                verbose=1
            ),
            CSVLogger(f'{save_path}/contract_test/0_2/train_log_CO_final.csv', append=True)
        ]

        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=2,
            epochs=200,
            callbacks=callbacks,
            verbose=1
        )

    except Exception as e:
        with open(f'{save_path}/error_final.txt', 'a') as f:
            f.write(f"\n{str(e)}")
