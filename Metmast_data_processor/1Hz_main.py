__author__ = 'Elliot I. Simon'
__email__ = 'ellsim@dtu.dk'
__version__ = 'October 20, 2021'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import xarray as xr
import datetime
from scipy.integrate import quad


def init(infile=sys.argv[1], outpath=sys.argv[2], path_8Hz=sys.argv[3]):
    #print(Path(infile), Path(outpath), Path(path_8Hz))
    return Path(infile), Path(outpath), Path(path_8Hz)


def load_files(infile, vane_32m_dir_offset):
    infile = [infile]
    # Keep the channel header but skip the initialization info and units
    rowlist = [0,2,3]
    df = pd.concat(map(lambda f: pd.read_csv(f, skiprows=rowlist, header=0, na_values='NAN', encoding='unicode-escape', on_bad_lines='skip'), infile))
    # Set the index as our datetime stamp
    df.set_index(pd.to_datetime(df['TIMESTAMP']), inplace=True)
    del df['TIMESTAMP']
    # Convert objects to correct dtypes
    df = df.convert_dtypes()
    # Map dict of status flag. Replace M values with True. Join back to original df and drop string columns
    status_dict = {'M': True, '': False, 'NA': False, 'NAN': False, np.nan: False}
    df = df.join(df.select_dtypes(exclude=['object', 'int', 'float']).replace(status_dict).fillna(False), lsuffix='_DROP')
    df = df.loc[:,~df.columns.str.contains('DROP')]
    df = df.sort_index()

    # Delete channels that we don't want, or will be joined from the reprocessed 8Hz data

    # Time-averaged status flags make no sense
    del df['USA3D_92m_S_Status_8Hz_Raw']
    del df['USA3D_32m_S_Status_8Hz_Raw']
    # 87m vane and geovane have quality issues
    del df['Vane_87m_NW_Dir_1Hz_Raw']
    del df['GeoV_87m_NW_Dir_1Hz_Raw']
    del df['GeoV_87m_NW_Temp_1Hz_Raw']
    del df['GeoV_87m_NW_Status_1Hz_Raw']
    # Voltage signal was not connected
    del df['SW12Volts(1)']
    # Sonic anemometer channels will be replaced with reprocessed 8Hz data
    del df['USA3D_92m_S_u_8Hz_Raw_Avg']
    del df['USA3D_92m_S_v_8Hz_Raw_Avg']
    del df['USA3D_92m_S_w_8Hz_Raw_Avg']
    del df['USA3D_92m_S_Spd_8Hz_Raw_Avg']
    del df['USA3D_92m_S_Dir_8Hz_Raw_Avg']
    del df['USA3D_92m_S_Dir_8Hz_Raw_Std']
    del df['USA3D_92m_S_Inc_8Hz_Raw_Avg']
    del df['USA3D_92m_S_Inc_8Hz_Raw_Std']
    del df['USA3D_92m_S_FloTC_8Hz_Raw_Avg']
    del df['USA3D_32m_S_u_8Hz_Raw_Avg']
    del df['USA3D_32m_S_v_8Hz_Raw_Avg']
    del df['USA3D_32m_S_w_8Hz_Raw_Avg']
    del df['USA3D_32m_S_Spd_8Hz_Raw_Avg']
    del df['USA3D_32m_S_Dir_8Hz_Raw_Avg']
    del df['USA3D_32m_S_Dir_8Hz_Raw_Std']
    del df['USA3D_32m_S_Inc_8Hz_Raw_Avg']
    del df['USA3D_32m_S_Inc_8Hz_Raw_Std']
    del df['USA3D_32m_S_FloTC_8Hz_Raw_Avg']

    # Correct 32m vane direction offset
    df['Vane_32m_NW_Dir_1Hz_Raw'] = (df['Vane_32m_NW_Dir_1Hz_Raw'] + vane_32m_dir_offset + 360) % 360

    return df


def cup_format(df):
    # Apply cup anemometer calibration gains/offsets. Convert from counts to m/s. Filter out very high values

    Aneo_96m_N = {'slope': 0.04590, 'offset': 0.2420}
    Aneo_96m_S = {'slope': 0.04592, 'offset': 0.2344}
    Aneo_92m_NW = {'slope': 0.04586, 'offset': 0.2568}
    Aneo_70m_NW = {'slope': 0.04594, 'offset': 0.2271}
    Aneo_48m_NW = {'slope': 0.04601, 'offset': 0.2216}
    Aneo_32m_NW = {'slope': 0.04603, 'offset': 0.2232}
    max_speed_filter = 40

    # Make new channels ending in _Calc and delete Raw data
    df['Aneo_96m_N_Spd_1Hz_Calc'] = (df['Aneo_96m_N_Spd_1Hz_Raw'] * Aneo_96m_N['slope']) + Aneo_96m_N['offset']
    df['Aneo_96m_S_Spd_1Hz_Calc'] = (df['Aneo_96m_S_Spd_1Hz_Raw'] * Aneo_96m_S['slope']) + Aneo_96m_S['offset']
    df['Aneo_92m_NW_Spd_1Hz_Calc'] = (df['Aneo_92m_NW_Spd_1Hz_Raw'] * Aneo_92m_NW['slope']) + Aneo_92m_NW['offset']
    df['Aneo_70m_NW_Spd_1Hz_Calc'] = (df['Aneo_70m_NW_Spd_1Hz_Raw'] * Aneo_70m_NW['slope']) + Aneo_70m_NW['offset']
    df['Aneo_48m_NW_Spd_1Hz_Calc'] = (df['Aneo_48m_NW_Spd_1Hz_Raw'] * Aneo_48m_NW['slope']) + Aneo_48m_NW['offset']
    df['Aneo_32m_NW_Spd_1Hz_Calc'] = (df['Aneo_32m_NW_Spd_1Hz_Raw'] * Aneo_32m_NW['slope']) + Aneo_32m_NW['offset']

    del df['Aneo_96m_N_Spd_1Hz_Raw']
    del df['Aneo_96m_S_Spd_1Hz_Raw']
    del df['Aneo_92m_NW_Spd_1Hz_Raw']
    del df['Aneo_70m_NW_Spd_1Hz_Raw']
    del df['Aneo_48m_NW_Spd_1Hz_Raw']
    del df['Aneo_32m_NW_Spd_1Hz_Raw']

    df['Aneo_96m_N_Spd_1Hz_Calc'].mask(df['Aneo_96m_N_Spd_1Hz_Calc'] > max_speed_filter, other=np.nan, errors='ignore',
                                       inplace=True)
    df['Aneo_96m_S_Spd_1Hz_Calc'].mask(df['Aneo_96m_S_Spd_1Hz_Calc'] > max_speed_filter, other=np.nan, errors='ignore',
                                       inplace=True)
    df['Aneo_92m_NW_Spd_1Hz_Calc'].mask(df['Aneo_92m_NW_Spd_1Hz_Calc'] > max_speed_filter, other=np.nan,
                                        errors='ignore', inplace=True)
    df['Aneo_70m_NW_Spd_1Hz_Calc'].mask(df['Aneo_70m_NW_Spd_1Hz_Calc'] > max_speed_filter, other=np.nan,
                                        errors='ignore', inplace=True)
    df['Aneo_48m_NW_Spd_1Hz_Calc'].mask(df['Aneo_48m_NW_Spd_1Hz_Calc'] > max_speed_filter, other=np.nan,
                                        errors='ignore', inplace=True)
    df['Aneo_32m_NW_Spd_1Hz_Calc'].mask(df['Aneo_32m_NW_Spd_1Hz_Calc'] > max_speed_filter, other=np.nan,
                                        errors='ignore', inplace=True)

    # Data logger SDM failure was repaired on December 2 (00:00). Cup anemometer data before this time is not correct
    # Set data before this cutoff to NaN

    if df.last_valid_index() < pd.to_datetime('2021-12-02 00:00:00', format='%Y-%m-%d %H:%M:%S'):
        df['Aneo_96m_N_Spd_1Hz_Calc'].loc[:] = np.nan
        df['Aneo_96m_S_Spd_1Hz_Calc'].loc[:] = np.nan
        df['Aneo_92m_NW_Spd_1Hz_Calc'].loc[:] = np.nan
        df['Aneo_70m_NW_Spd_1Hz_Calc'].loc[:] = np.nan
        df['Aneo_48m_NW_Spd_1Hz_Calc'].loc[:] = np.nan
        df['Aneo_32m_NW_Spd_1Hz_Calc'].loc[:] = np.nan

    return df

def join_8Hz(df, path_8Hz):
    # Join pre-processed 8Hz -> 1Hz downsampled data
    # Logic to find and load corresponding periods of 8Hz data
    start = df.first_valid_index()
    end = df.last_valid_index()
    #print(start, end)

    files_8Hz = sorted(path_8Hz.glob('*.nc'))

    timestamp = []
    for f in files_8Hz:
        timestamp.append(str(f).split('8Hz_')[-1].split('.nc')[0])

    df_8Hz_files = pd.DataFrame(timestamp, columns=['timestamp'])

    df_8Hz_files[['start', 'end']] = df_8Hz_files['timestamp'].str.split('_', expand=True)
    df_8Hz_files['start'] = pd.to_datetime(df_8Hz_files['start'], format='%Y-%m-%d-%H-%M-%S', errors='coerce')
    df_8Hz_files['end'] = pd.to_datetime(df_8Hz_files['end'], format='%Y-%m-%d-%H-%M-%S', errors='coerce')
    df_8Hz_files['filename'] = files_8Hz
    del df_8Hz_files['timestamp']

    # Find corresponding files that match current period
    indices = df_8Hz_files[(df_8Hz_files['start'] >= start - pd.Timedelta('10Min')) &
                           (df_8Hz_files['end'] <= end + pd.Timedelta('10Min'))].index
    indices = indices.insert(0, indices[0] - 1)
    indices = indices.insert(len(indices), indices[-1] + 1)
    # print(indices)

    # Ensure we don't go out of bounds at the end
    if any(i > df_8Hz_files.index[-1] for i in indices):
        indices = indices[0:-1]
        print('Reached end of 8Hz files')

    # Make list of files to load
    files_8Hz_load = list(df_8Hz_files.iloc[indices]['filename'].values)

    # Load the file list, resample and join channels of interest
    ds_8Hz = xr.open_mfdataset(files_8Hz_load, chunks='auto')

    df_8Hz = ds_8Hz.to_pandas()
    df_8Hz = df_8Hz.resample('1S', label='right').mean()

    del df_8Hz['RECORD']
    del df_8Hz['USA3D_92m_S_Status_8Hz_Raw']
    del df_8Hz['USA3D_32m_S_Status_8Hz_Raw']

    new_columns = {column: column + "_Avg" for column in df_8Hz.columns}
    df_8Hz.rename(columns=new_columns, inplace=True)
    
    # Repeat sonic reconstruction following resampling (otherwise direction averaging near 0 degrees is incorrect)

    del df_8Hz['USA3D_92m_S_Spd_8Hz_Calc_Avg']
    del df_8Hz['USA3D_32m_S_Spd_8Hz_Calc_Avg']
    del df_8Hz['USA3D_92m_S_Dir_8Hz_Calc_Avg']
    del df_8Hz['USA3D_32m_S_Dir_8Hz_Calc_Avg']

    # Re-do reconstruction of sonic speed and direction
    # 3D reconstuction (x,y,z)

    df_8Hz['USA3D_92m_S_Spd_8Hz_Calc_Avg'] = np.sqrt(df_8Hz['USA3D_92m_S_x_8Hz_Raw_Avg']**2 + df_8Hz['USA3D_92m_S_y_8Hz_Raw_Avg']**2 + df_8Hz['USA3D_92m_S_z_8Hz_Raw_Avg']**2)
    df_8Hz['USA3D_32m_S_Spd_8Hz_Calc_Avg'] = np.sqrt(df_8Hz['USA3D_32m_S_x_8Hz_Raw_Avg']**2 + df_8Hz['USA3D_32m_S_y_8Hz_Raw_Avg']**2 + df_8Hz['USA3D_32m_S_z_8Hz_Raw_Avg']**2)

    df_8Hz['USA3D_92m_S_Dir_8Hz_Calc_Avg'] = (np.rad2deg(np.arctan2(df_8Hz['USA3D_92m_S_y_8Hz_Raw_Avg'], df_8Hz['USA3D_92m_S_x_8Hz_Raw_Avg']))+360)%360
    df_8Hz['USA3D_32m_S_Dir_8Hz_Calc_Avg'] = (np.rad2deg(np.arctan2(df_8Hz['USA3D_32m_S_y_8Hz_Raw_Avg'], df_8Hz['USA3D_32m_S_x_8Hz_Raw_Avg']))+360)%360

    # Apply offset to wind direction

    sonic_32m_dir_offset = 208.52
    sonic_92m_dir_offset = 211.66

    df_8Hz['USA3D_32m_S_Dir_8Hz_Calc_Avg'] = (df_8Hz['USA3D_32m_S_Dir_8Hz_Calc_Avg']+sonic_32m_dir_offset+360)%360
    df_8Hz['USA3D_92m_S_Dir_8Hz_Calc_Avg'] = (df_8Hz['USA3D_92m_S_Dir_8Hz_Calc_Avg']+sonic_92m_dir_offset+360)%360

    df = df.join(df_8Hz, how='left')
    return df


def sst_sky_correction(df):
    # Apply SST sky correction
    # Credit for method and formulas to Richard Frühmann at UL (Richard.Fruehmann@ul.com)

    # Create calibration curve for IR data
    lambda1 = 8e-6  # lower spectral limit
    lambda2 = 14e-6  # upper spectral limit
    Kplanck = 6.626e-34  # Planck constant
    Kboltz = 1.38e-23  # Boltzmann constant
    Klight = 299.8e6  # Speed of light
    Tvir = np.arange(173.15, 373.15 + 0.05, 0.05)
    # Temperature range from -100 to +100 °C (calibration range) in Kelvin.
    # Our data has 0.1 degree precision, but since we start at .15 then we should do 0.05 steps to minimize rounding errors.

    def integrand(x):
        # print(x, Tvir[i])
        result = (2 * Kplanck * Klight ** 2) / ((x ** 5) * (np.exp(Kplanck * Klight / (x * Kboltz * Tvir[i])) - 1))
        return result

    xi = np.zeros(shape=(len(Tvir), 1))
    for i, temp in enumerate(Tvir):
        # print(i,temp)
        xi[i] = quad(integrand, lambda1, lambda2)[0]
        # Spectral Irradiance. Take only integration result and not error estimate

    # Make DataFrame of the blackbody equivalent radiant emission curve
    df_bb = pd.DataFrame(xi)
    df_bb.columns = ['RadEm']
    df_bb['degK'] = np.round(Tvir, 2)

    def lookup_RadEm(degK):
        '''Apply precalculated radiant emission curve lookup table to measured temperature values
        Our measurement data has a precision of 0.1 degrees Kelvin.
        This takes a while to run!'''
        match = df_bb['degK'] == np.round(degK, 1)
        RadEm = df_bb['RadEm'][match]
        return RadEm.values[0]

    # Assume the Sky acts as black body, don't calculate emissivity for the Sky channel
    df['SST01_Sea_1Hz_RadEm_Calc'] = df['SST01_Sea_MD_1Hz_Raw'].apply(lookup_RadEm)

    df['SST_SkyCorrection_1Hz_Calc'] = (df['SST01_Sea_MD_1Hz_Raw'] - (1 - df['SST01_Sea_1Hz_RadEm_Calc']) *
                                        df['SST01_Sky_MD_1Hz_Raw']) / df['SST01_Sea_1Hz_RadEm_Calc']

    return df

def convert_xarray(df):
    # Convert pandas._libs.missing.NAType to np.nan
    df.fillna(np.nan, inplace=True)

    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(df)

    # Ensure missing data is correct format for netcdf file (must be np.nan (float object) and not pandas.NaT)
    ds = ds.fillna(np.nan)

    # Cast xarray variables into correct dtypes (unsure why not preserved when going pandas -> xarray)
    for c in df.columns:
        if c in ['RECORD', 'GeoV_87m_NW_Status_1Hz_Raw', 'Pres_96m_PR_1Hz_Raw', 'Pres_C_PR_1Hz_Raw']:
            ds[c] = ds[c].astype(np.int64)
        else:
            ds[c] = ds[c].astype(np.float64)

    return df, ds


def apply_attributes(ds):
    # Add attributes (units, channel description, etc)
    ds.attrs['Author'] = 'Elliot Simon'
    ds.attrs['Contact'] = 'ellsim@dtu.dk'
    ds.attrs['Project'] = 'OWA GLOBE'
    ds.attrs['Position_UTM'] = '411734.4371 m E, 6028967.271 m N, UTM WGS84 Zone32'
    ds.attrs['Time_Zone'] = 'UTC, no DST shift'
    ds.attrs['NTP_Sync_Enabled'] = 'True'
    ds.attrs['Date_Processed'] = str(datetime.datetime.now())

    ds['RECORD'].attrs['description'] = 'Sample index'
    ds['RECORD'].attrs['units'] = 'Integer index'
    ds['RECORD'].attrs['height'] = 'N/A'

    ds['Temp_96m_TC_1Hz_Raw'].attrs['description'] = 'Air temperature at 96m height'
    ds['Temp_96m_TC_1Hz_Raw'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_96m_TC_1Hz_Raw'].attrs['height'] = '96m'

    ds['Temp_CR_TC_1Hz_Raw'].attrs['description'] = 'Air temperature at container roof position, 24.66m height'
    ds['Temp_CR_TC_1Hz_Raw'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_CR_TC_1Hz_Raw'].attrs['height'] = '24.66m'

    ds['TempD_CR_96m_dTC_1Hz_Raw'].attrs[
        'description'] = 'Air temperature difference between 96m and container roof (24.66m)'
    ds['TempD_CR_96m_dTC_1Hz_Raw'].attrs['units'] = 'Degrees Celsius'
    ds['TempD_CR_96m_dTC_1Hz_Raw'].attrs['height'] = '96m'

    ds['RHS_96m_RH_1Hz_Raw'].attrs['description'] = 'Relative humidity at 96m height'
    ds['RHS_96m_RH_1Hz_Raw'].attrs['units'] = 'Percent'
    ds['RHS_96m_RH_1Hz_Raw'].attrs['height'] = '96m'

    ds['RHS_CR_RH_1Hz_Raw'].attrs['description'] = 'Relative humidity at container roof position, 24.66m height'
    ds['RHS_CR_RH_1Hz_Raw'].attrs['units'] = 'Percent'
    ds['RHS_CR_RH_1Hz_Raw'].attrs['height'] = '24.66m'

    ds['Pres_96m_PR_1Hz_Raw'].attrs['description'] = 'Air pressure at 96m height'
    ds['Pres_96m_PR_1Hz_Raw'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_96m_PR_1Hz_Raw'].attrs['height'] = '96m'

    ds['Pres_C_PR_1Hz_Raw'].attrs['description'] = 'Air pressure at container position, 22.3m height'
    ds['Pres_C_PR_1Hz_Raw'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_C_PR_1Hz_Raw'].attrs['height'] = '22.3m'

    ds['SST01_Sky_MD_1Hz_Raw'].attrs['description'] = 'Infrared radiometer reading, pointed up at the sky'
    ds['SST01_Sky_MD_1Hz_Raw'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sky_MD_1Hz_Raw'].attrs['height'] = '20m'

    ds['SST01_Sea_MD_1Hz_Raw'].attrs['description'] = 'Infrared radiometer reading, pointed down at the sea surface'
    ds['SST01_Sea_MD_1Hz_Raw'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sea_MD_1Hz_Raw'].attrs['height'] = '20m'

    ds['SST01_Sea_1Hz_RadEm_Calc'].attrs[
        'description'] = 'Radiant emission calculated from the infrared radiometer pointed down at sea surface using a calculated spectral radiance curve applied using a lookup table'
    ds['SST01_Sea_1Hz_RadEm_Calc'].attrs['units'] = 'Watt/steradian/square meter'
    ds['SST01_Sea_1Hz_RadEm_Calc'].attrs['height'] = '20m'

    ds['SST_SkyCorrection_1Hz_Calc'].attrs[
        'description'] = 'Sea surface temperature with sky correction method applied to correct for background radiation'
    ds['SST_SkyCorrection_1Hz_Calc'].attrs['units'] = 'Degrees Kelvin'
    ds['SST_SkyCorrection_1Hz_Calc'].attrs['height'] = '20m'

    ds['Vane_32m_NW_Dir_1Hz_Raw'].attrs[
        'description'] = 'Wind vane (North-West boom) wind direction at 32m height with offset applied'
    ds['Vane_32m_NW_Dir_1Hz_Raw'].attrs['units'] = 'Degrees'
    ds['Vane_32m_NW_Dir_1Hz_Raw'].attrs['height'] = '32m'

    ds['Aneo_96m_N_Spd_1Hz_Calc'].attrs[
        'description'] = 'Cup anemometer (North boom) wind speed at 96m height with calibration gains/offsets applied'
    ds['Aneo_96m_N_Spd_1Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_N_Spd_1Hz_Calc'].attrs['height'] = '96m'

    ds['Aneo_96m_S_Spd_1Hz_Calc'].attrs[
        'description'] = 'Cup anemometer (South boom) wind speed at 96m height with calibration gains/offsets applied'
    ds['Aneo_96m_S_Spd_1Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_S_Spd_1Hz_Calc'].attrs['height'] = '96m'

    ds['Aneo_92m_NW_Spd_1Hz_Calc'].attrs[
        'description'] = 'Cup anemometer (North-West boom) wind speed at 92m height with calibration gains/offsets applied'
    ds['Aneo_92m_NW_Spd_1Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['Aneo_92m_NW_Spd_1Hz_Calc'].attrs['height'] = '92m'

    ds['Aneo_70m_NW_Spd_1Hz_Calc'].attrs[
        'description'] = 'Cup anemometer (North-West boom) wind speed at 70m height with calibration gains/offsets applied'
    ds['Aneo_70m_NW_Spd_1Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['Aneo_70m_NW_Spd_1Hz_Calc'].attrs['height'] = '70m'

    ds['Aneo_48m_NW_Spd_1Hz_Calc'].attrs[
        'description'] = 'Cup anemometer (North-West boom) wind speed at 48m height with calibration gains/offsets applied'
    ds['Aneo_48m_NW_Spd_1Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['Aneo_48m_NW_Spd_1Hz_Calc'].attrs['height'] = '48m'

    ds['Aneo_32m_NW_Spd_1Hz_Calc'].attrs[
        'description'] = 'Cup anemometer (North-West boom) wind speed at 32m height with calibration gains/offsets applied'
    ds['Aneo_32m_NW_Spd_1Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['Aneo_32m_NW_Spd_1Hz_Calc'].attrs['height'] = '32m'

    ds['USA3D_92m_S_x_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) x-axis speed at 92m height. Outliers removed using moving-median filter. Downsampled to 1Hz'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_92m_S_y_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) y-axis speed at 92m height. Outliers removed using moving-median filter. Downsampled to 1Hz'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_92m_S_z_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) z-axis speed at 92m height. Outliers removed using moving-median filter. Downsampled to 1Hz'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_x_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) x-axis speed at 32m height. Outliers removed using moving-median filter. Downsampled to 1Hz'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_32m_S_y_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) y-axis speed at 32m height. Outliers removed using moving-median filter. Downsampled to 1Hz'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_32m_S_z_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) z-axis speed at 32m height. Outliers removed using moving-median filter. Downsampled to 1Hz'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) temperature at 92m height. Downsampled to 1Hz'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) temperature at 32m height. Downsampled to 1Hz'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 92m height, raw output. Downsampled to 1Hz'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 32m height, raw output. Downsampled to 1Hz'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 92m height calculated using: arctan2(z,sqrt(x^2+y^2)). Downsampled to 1Hz'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 32m height calculated using: arctan2(z,sqrt(x^2+y^2)). Downsampled to 1Hz'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind speed at 92m height calculated using 3D reconstruction: sqrt(x^2+y^2+z^2). Downsampled to 1Hz'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind speed at 32m height calculated using 3D reconstruction: sqrt(x^2+y^2+z^2). Downsampled to 1Hz'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind direction at 92m height calculated using 2D reconstruction: (rad2deg(arctan2(y,x))+360)%360 with offset applied. Downsampled to 1Hz'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind direction at 32m height calculated using 2D reconstruction: (rad2deg(arctan2(y,x))+360)%360 with offset applied. Downsampled to 1Hz'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg'].attrs['height'] = '32m'

    return ds


def save_disk(df, ds, outpath):
    # Make dict for compression encoding
    # Example, raw data files = 745 MB. No compression = 980 MB. zlib compression = 628 MB
    encode_dict = {
        'RECORD': {'zlib': True},
        'Vane_32m_NW_Dir_1Hz_Raw': {'zlib': True},
        'Temp_96m_TC_1Hz_Raw': {'zlib': True},
        'Temp_CR_TC_1Hz_Raw': {'zlib': True},
        'RHS_96m_RH_1Hz_Raw': {'zlib': True},
        'RHS_CR_RH_1Hz_Raw': {'zlib': True},
        'Pres_96m_PR_1Hz_Raw': {'zlib': True},
        'Pres_C_PR_1Hz_Raw': {'zlib': True},
        'TempD_CR_96m_dTC_1Hz_Raw': {'zlib': True},
        'SST01_Sky_MD_1Hz_Raw': {'zlib': True},
        'SST01_Sea_MD_1Hz_Raw': {'zlib': True},
        'Aneo_96m_N_Spd_1Hz_Calc': {'zlib': True},
        'Aneo_96m_S_Spd_1Hz_Calc': {'zlib': True},
        'Aneo_92m_NW_Spd_1Hz_Calc': {'zlib': True},
        'Aneo_70m_NW_Spd_1Hz_Calc': {'zlib': True},
        'Aneo_48m_NW_Spd_1Hz_Calc': {'zlib': True},
        'Aneo_32m_NW_Spd_1Hz_Calc': {'zlib': True},
        'USA3D_92m_S_x_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_92m_S_y_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_92m_S_z_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_92m_S_FloTC_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_32m_S_x_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_32m_S_y_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_32m_S_z_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_32m_S_FloTC_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Raw_Avg': {'zlib': True},
        'USA3D_92m_S_Spd_8Hz_Calc_Avg': {'zlib': True},
        'USA3D_32m_S_Spd_8Hz_Calc_Avg': {'zlib': True},
        'USA3D_92m_S_Dir_8Hz_Calc_Avg': {'zlib': True},
        'USA3D_32m_S_Dir_8Hz_Calc_Avg': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Calc_Avg': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Calc_Avg': {'zlib': True},
        'SST01_Sea_1Hz_RadEm_Calc': {'zlib': True},
        'SST_SkyCorrection_1Hz_Calc': {'zlib': True}
    }

    fname = 'NSO-met-mast-data-1Hz_' + str(df.first_valid_index().strftime('%Y-%m-%d-%H-%M-%S')) + '_' + str(
        df.last_valid_index().strftime('%Y-%m-%d-%H-%M-%S'))
    print('..Writing to disk..')
    ds.to_netcdf(str(outpath) + '\\' + fname + '.nc', format='NETCDF4', engine='netcdf4', encoding=encode_dict)
    print(fname)
    return fname


def main():
    vane_32m_dir_offset = 150.49

    infile, outpath, path_8Hz = init()
    df = load_files(infile, vane_32m_dir_offset)
    df = cup_format(df)
    df = join_8Hz(df, path_8Hz)
    df = sst_sky_correction(df)
    # Correct geovane??
    # Wind vane offsets??
    #print(df)
    df, ds = convert_xarray(df)
    ds = apply_attributes(ds)
    fname = save_disk(df, ds, outpath)

    print('Done!')


if __name__ == '__main__':
    main()
