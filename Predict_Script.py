"""
=============================================================
预测主脚本（适配仅权重文件的 Surface Inference 模型）
=============================================================
"""
import os
import time
import numpy as np
import xarray as xr
import netCDF4

from model_UNet_AG_PhysLoss import *
from extract_2pre_day_CO import *
from extract_2pre_day_NO2 import *
from output_mean import *
from tensorflow.keras.layers.experimental.preprocessing import Normalization

if __name__ == "__main__":
    try:
        # ========== 初始设置 ==========
        start_time = time.time()
        path = os.getenv('WorkDir')
        startdate = os.getenv('SDateDir')
        enddate = int(startdate) + 100

        # 读取经纬度
        nc_ref = xr.open_dataset(
            "/share/home/fengsz1/zhuanghl/RAPASv2_ensrf/ensrfv2/working/2024123100/WRFCMAQ/42/WRFCMAQ_27km_hourly_20241231.nc",
            engine="netcdf4"
        )

        xlon = np.expand_dims(nc_ref.XLON[0, 0].values, axis=(0, 1))  # (1,1,ROW,COL)
        xlat = np.expand_dims(nc_ref.XLAT[0, 0].values, axis=(0, 1))

        # ========== CO ==========
        co_x = data_assemble_step1(path)
        print(co_x.shape)
        # model_path = f"/share/home/fengsz1/zhuanghl/AGT_UNet/Final_model_COandNOx/COmodel/CO_model_073_0.0236.keras"
        # norm_layer = Normalization(axis=-1)
        # norm_layer.build((None, 159, 219, co_x.shape[-1]))


        # model_co = CustomUNet(
        #     input_shape=(159, 219, co_x.shape[-1]),
        #     norm_layer=norm_layer,
        #     dropout_rate=0.0
        # ).build_model()
        # model_co.load_weights(model_path)

        # norm_layer.mean.assign(np.load("/share/home/fengsz1/zhuanghl/AGT_UNet/Final_model_COandNOx/COmodel/norm_mean.npy"))
        # norm_layer.variance.assign(np.load("/share/home/fengsz1/zhuanghl/AGT_UNet/Final_model_COandNOx/COmodel/norm_var.npy"))

        # co_y = model_co.predict(co_x, batch_size=1, verbose=1)
        # co_y = np.transpose(co_y, (0, 3, 1, 2))

        # # ========== NO2 ==========
        # no2_x = data_assemble_no2(path)
        # model_path_no2 = f"/share/home/fengsz1/zhuanghl/AGT_UNet/Final_model_COandNOx/NO2model/CO_model_065_0.0020.keras"
        # norm_layer_no2 = Normalization(axis=-1)
        # norm_layer_no2.build((None, 159, 219, no2_x.shape[-1]))

        # model_no2 = CustomUNet(
        #     input_shape=(159, 219, no2_x.shape[-1]),
        #     norm_layer=norm_layer_no2,
        #     dropout_rate=0.0
        # ).build_model()
        # model_no2.load_weights(model_path_no2)

        # norm_layer_no2.mean.assign(np.load("/share/home/fengsz1/zhuanghl/AGT_UNet/Final_model_COandNOx/NO2model/norm_mean.npy"))
        # norm_layer_no2.variance.assign(np.load("/share/home/fengsz1/zhuanghl/AGT_UNet/Final_model_COandNOx/NO2model/norm_var.npy"))

        # no2_y = model_no2.predict(no2_x, batch_size=1, verbose=1)
        # no2_y = np.transpose(no2_y, (0, 3, 1, 2))
        # # with open(f'/share/home/fengsz1/zhuanghl/output.txt', 'a') as f:
        # #     f.write(f"\n预测格式：x_test={x_test.shape}\n")

        # # with open(f'/share/home/fengsz1/zhuanghl/output.txt', 'a') as f:
        # #     f.write(f"y_pred 维度: {y_pred.shape}\n")

        # # ========== 输出 NetCDF ==========
        # WRFCMAQ_path = os.path.join(path, 'WRFCMAQ')
        # os.makedirs(WRFCMAQ_path, exist_ok=True)

        # sd = int(startdate)
        # for i in range(41):
        #     folder_name = os.path.join(WRFCMAQ_path, f"{i + 1:02d}")
        #     os.makedirs(folder_name, exist_ok=True)

        #     # TFLAG
        #     tflag_data = tflag_num(str(startdate))

        #     # CO 数据
        #     co_data = co_y[i]  # shape (1, 159, 219)
        #     if co_data.ndim == 3:
        #         co_data = np.expand_dims(co_data, axis=1)  # -> (1, 1, 159, 219)

        #     # NO2 数据
        #     no2_data = no2_y[i]  # shape (1, 159, 219)
        #     if no2_data.ndim == 3:
        #         no2_data = np.expand_dims(no2_data, axis=1)  # -> (1, 1, 159, 219)

        #     # 构建 Dataset
        #     TSTEP = 1
        #     LAY = 1
        #     ROW = 159
        #     COL = 219
        #     ds = xr.Dataset(
        #         {
        #             "TFLAG": (["TSTEP", "VAR", "DATE-TIME"], tflag_data, {
        #                 "units": "<YYYYDDD,HHMMSS>",
        #                 "long_name": "TFLAG",
        #                 "var_desc": "Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS"
        #             }),
        #             "CO": (["TSTEP", "LAY", "ROW", "COL"], co_data.astype(np.float32), {
        #                 "long_name": 'CO',
        #                 "units": 'ppmV',
        #                 "var_desc": "CO[1]"
        #             }),
        #             "NO2": (["TSTEP", "LAY", "ROW", "COL"], no2_data.astype(np.float32), {
        #                 "long_name": 'NO2',
        #                 "units": 'ppmV',
        #                 "var_desc": "NO2[1]"
        #             }),                    
        #             "XLON": (["TSTEP", "LAY", "ROW", "COL"], np.broadcast_to(xlon, co_data.shape), {
        #                 "long_name": 'XLON',
        #                 "units": 'DEGREES',
        #                 "var_desc": 'LON[2]'
        #             }),
        #             "XLAT": (["TSTEP", "LAY", "ROW", "COL"], np.broadcast_to(xlat, co_data.shape), {
        #                 "long_name": "XLAT",
        #                 "units": "DEGREES",
        #                 "var_desc": "LAT[2]"
        #             })
        #         }
        #     )

        #     # 写出时确保不添加其他维度变量
        #     encoding = {
        #         "TFLAG": {"_FillValue": None},
        #         "CO": {"_FillValue": -9.999e36},
        #         "NO2": {"_FillValue": -9.999e36},
        #         "XLON": {"_FillValue": -9.999e36},
        #         "XLAT": {"_FillValue": -9.999e36}
        #     }
        #     nc_file_path = os.path.join(folder_name, f"WRFCMAQ_27km_hourly_{str(sd // 100)}.nc")
        #     ds.to_netcdf(nc_file_path, encoding=encoding)

        # # ========== 完成 ==========
        # end_time = time.time()

        # with open(f"/share/home/fengsz1/zhuanghl/output.txt", "a") as f:
        #     f.write(f"总用时：{end_time - start_time:.2f} 秒\n")

    except Exception as e:
        with open(f"/share/home/fengsz1/zhuanghl/error.txt", "a") as f:
            f.write(f"{str(e)}\n")