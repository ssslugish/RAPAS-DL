import xarray as xr
import numpy as np
import os
from netCDF4 import Dataset
#from scipy.spatial import cKDTree
import pandas as pd
#from scipy.interpolate import RegularGridInterpolator
import math

#st = pd.read_csv("/share/home/fengsz1/zhuanghl/AGT_UNet/station_list_202202.csv")
#target_lats = st['lat']
#target_lons = st['lon']

def get_bcon(bcon_path):
    bcon = Dataset(bcon_path)

    # 网格大小
    NY = 159
    NX = 219

    # 读取 CO 边界（TSTEP0, LAY0）
    co_perim = (bcon.variables["CO"][:, 0, :].mean(axis=0))

    # 分四段
    west = co_perim[0:NY]
    east = co_perim[NY:2 * NY]
    south = co_perim[2 * NY:2 * NY + NX]
    north = co_perim[2 * NY + NX:2 * (NY + NX)]

    # 构造 2D 场
    BC_field = np.zeros((NY, NX))

    # 写入边界
    BC_field[:, 0] = west  # 西
    BC_field[:, -1] = east  # 东
    BC_field[0, :] = south  # 南
    BC_field[-1, :] = north  # 北

    return BC_field

# def get_waccm(waccm_path):
#     waccm = xr.open_dataset(waccm_path)
#     co_waccm = waccm['CO'].isel(lev=0)  # (time, 59, 89)
#     lat_w = waccm['lat'].values  # (59,)
#     lon_w = waccm['lon'].values  # (89,)

#     wrf = xr.open_dataset("/share/home/fengsz1/zhuanghl/RAPASv2_ensrf/ensrfv2/working/2021122600/ibcon/42/WRFCMAQ_27km_hourly_20211225.nc")
#     lats_cmaq = wrf['XLAT'][0, 0].values
#     lons_cmaq = wrf['XLON'][0, 0].values

#     co_day = co_waccm[:,:,:].mean(axis=0).values
#     interp_func = RegularGridInterpolator(
#         (lat_w, lon_w),
#         co_day,
#         bounds_error=False,
#         fill_value=np.nan
#     )

#     points = np.array([lats_cmaq.flatten(), lons_cmaq.flatten()]).T
#     co_interp = interp_func(points).reshape(159, 219)
#     return co_interp

# def get_station_mask(
#         nc_file_path,
#         target_lats=target_lats,
#         target_lons=target_lons,
#         lat_var='XLAT',
#         lon_var='XLONG'
# ):
#     # 读取数据
#     ds = Dataset(nc_file_path, mode='r')

#     # 经纬度 (假设 shape 为 (Time, Layer, ROW, COL)，只取第一个时间点 & 层)
#     lats = ds.variables[lat_var][0, 3:162, 3:222]  # shape: (152, 216)
#     lons = ds.variables[lon_var][0, 3:162, 3:222]  # shape: (152, 216)

#     ds.close()

#     H, W = lats.shape

#     # 构建 KDTree 进行最近点匹配
#     lat_flat = lats.flatten()
#     lon_flat = lons.flatten()
#     grid_points = np.column_stack((lat_flat, lon_flat))  # shape: (152*216, 2)
#     tree = cKDTree(grid_points)

#     target_points = np.column_stack((target_lats, target_lons))  # shape: (n, 2)
#     distances, indices = tree.query(target_points)

#     # 构建掩码数组
#     station_mask = np.zeros((H * W,), dtype=np.float32)

#     for idx in indices:
#         station_mask[idx] = 1.0  # 将匹配位置置为 1

#     station_mask = station_mask.reshape(H, W)  # (152, 216)
#     station_mask = np.expand_dims(station_mask, axis=(0, -1))  # (1, 152, 216, 1)

#     return station_mask

# def preprocess_emis(emis, q=99.9):
#     emis = np.clip(emis, 0, None)
#     vmax = np.percentile(emis, q)
#     emis = np.clip(emis, 0, vmax)
#     return emis

def data_extract_step1(wrf1_path, wrf2_path, ibcon_path, emis_path):
    icon_count = 0
    for file in os.listdir(ibcon_path):
        if file.startswith('BCON'):
            bcon_path = os.path.join(ibcon_path, file)
            bcon = get_bcon(bcon_path)

    if icon_count == 0:
        cmaq_first_path = os.path.join(ibcon_path, '42')
        for cmaq_first_file in os.listdir(cmaq_first_path):
            if cmaq_first_file.startswith('WRFCMAQ'):
                cmaq_firstfile_path = os.path.join(cmaq_first_path, cmaq_first_file)
                with xr.open_dataset(cmaq_firstfile_path, engine="netcdf4") as da:
                    icon = da.CO[:, 0].mean(axis=0)

    wrf1 = xr.open_dataset(wrf1_path, engine="netcdf4")
    wrf2 = xr.open_dataset(wrf2_path, engine="netcdf4")
    wrf_values = xr.concat([wrf1, wrf2], dim='Time')

    times_str = [t.decode('utf-8') for t in wrf1.Times.values]
    times_dt = pd.to_datetime(times_str, format="%Y-%m-%d_%H:%M:%S")
    unique_days = times_dt.normalize().unique()

    day_of_year = pd.to_datetime(unique_days).dayofyear
    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)
    doy_sin_grid = np.full((159, 219), doy_sin, dtype=np.float32)
    doy_cos_grid = np.full((159, 219), doy_cos, dtype=np.float32)

    dow = pd.to_datetime(unique_days).dayofweek  # 0=周一 ... 6=周日
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    dow_sin_grid = np.full((159, 219), dow_sin, dtype=np.float32)
    dow_cos_grid = np.full((159, 219), dow_cos, dtype=np.float32)

    LU_INDEX = wrf_values.LU_INDEX[0]
    LU_INDEX = LU_INDEX.isel(south_north=slice(3, 162), west_east=slice(3, 222))
    LU_INDEX_safe = np.clip(LU_INDEX.astype(int), 1, 24)
    lu_onehot = np.eye(24)[LU_INDEX_safe - 1]

    VEGFRA = wrf_values.VEGFRA.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    XLONG = wrf_values.XLONG[0]
    XLONG = XLONG.isel(south_north=slice(3, 162), west_east=slice(3, 222))

    XLAT = wrf_values.XLAT[0]
    XLAT = XLAT.isel(south_north=slice(3, 162), west_east=slice(3, 222))

    lon_sin = np.sin(XLONG * math.pi / 180)
    lon_cos = np.cos(XLONG * math.pi / 180)
    lat_sin = np.sin(XLAT * math.pi / 180)
    lat_cos = np.cos(XLAT * math.pi / 180)

    T2 = wrf_values.T2.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    PSFC = wrf_values.PSFC.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    Q2 = wrf_values.Q2.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    T_c = T2 - 273.15
    e = (Q2 * PSFC) / (0.622 + 0.378 * Q2)
    e_s = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5)) * 100
    RH = (e / e_s) * 100
    RH = np.clip(RH, 0, 100)

    U10 = wrf_values.U10.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))
    V10 = wrf_values.V10.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))
    WSPD = np.sqrt(U10 ** 2 + V10 ** 2)

    RAINC = wrf_values.RAINC.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))
    RAINNC = wrf_values.RAINNC.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))
    RAIN = RAINC + RAINNC

    PBLH = wrf_values.PBLH.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    HGT = wrf_values.HGT.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    SWDOWN = wrf_values.SWDOWN.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    ALBEDO = wrf_values.ALBEDO.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))

    HFX = wrf_values.HFX.mean(dim="Time").isel(south_north=slice(3, 162), west_east=slice(3, 222))
    VC = PBLH.values * WSPD.values

    x_data = []
    for emisday_list in os.listdir(emis_path):
        emisday_path = os.path.join(emis_path, emisday_list)
        for wrfout_file in os.listdir(emisday_path):
            if emisday_list.startswith('emission'):
                emisfile_path = os.path.join(emisday_path, wrfout_file)
                with xr.open_dataset(emisfile_path, engine="netcdf4") as ds:
                    emis_area = ds.area[0, 2].astype(np.float32)
                    emis_point = ds.point[0, 2].astype(np.float32)

                combined_data_1 = np.stack([
                    emis_area, emis_point, icon, bcon,
                    T2, PSFC, Q2, RH, U10, V10, WSPD,
                    RAIN, PBLH, VC, HGT, SWDOWN, HFX, ALBEDO, VEGFRA,
                    lat_cos, lat_sin, lon_cos, lon_sin,
                    doy_sin_grid, doy_cos_grid, dow_sin_grid, dow_cos_grid
                ], axis=-1)

                combined_data_1 = np.concatenate([combined_data_1, lu_onehot], axis=-1)

                x_data.append(combined_data_1)

    x_train = np.stack(x_data, axis=0)
    return x_train

def data_assemble_step1(path):
    wrf = []
    ibcon = []
    emis = []
    cmaq = []
    
    wrf_path = os.path.join(path, 'wrflist')
    for wrf_file in os.listdir(wrf_path):
        if wrf_file.startswith('wrfout_'):
            wrf_file_path = os.path.join(wrf_path, wrf_file)
            wrf.append(wrf_file_path)
            
    ibcon_path = os.path.join(path, 'ibcon')
    ibcon.append(ibcon_path)
    
    emis_path = os.path.join(path, 'old_en_emis')
    emis.append(emis_path)

    j = 0
    x_dataset = []
    for i in range(len(wrf) // 2):
        x = data_extract_step1(wrf[j], wrf[j + 1], ibcon[i], emis[i])
        x_dataset.append(x)
        j += 2

    x_test = np.concatenate(x_dataset[:], axis=0)

    return x_test