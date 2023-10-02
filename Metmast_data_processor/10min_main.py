__author__ = 'Elliot I. Simon'
__email__ = 'ellsim@dtu.dk'
__version__ = 'October 20, 2021'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import datetime
#from matplotlib import pyplot as plt


def init(infile=sys.argv[1], outpath=sys.argv[2], path_1Hz=sys.argv[3], path_iwes_92m=sys.argv[4],
         path_iwes_32m=sys.argv[5]):
    #print(Path(infile), Path(outpath), Path(path_8Hz))
    return Path(infile), Path(outpath), Path(path_1Hz), Path(path_iwes_92m), Path(path_iwes_32m)


def loadfiles(infile):
    '''
    Input: list of .DAT files from met-mast data logger.
    Output: formatted pandas DataFrame containing measurement data.
    '''
    # Keep the channel header but skip the initialization info and units
    rowlist = [0,2,3]
    # Convert windows Path to string filename
    infile = str(infile)
    # Load the data
    df = pd.read_csv(infile, skiprows=rowlist, header=0, na_values='NAN', encoding='unicode-escape', on_bad_lines='skip')
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
    return df


def format_df(df, path_1Hz, path_iwes_92m, path_iwes_32m):
    # Delete channels that we don't want, or will be joined from the reprocessed 1Hz data
    del_channels = ['Aneo_96m_N_Spd_1Hz_Raw_Avg', 'Aneo_96m_N_Spd_1Hz_Raw_Std', 'Aneo_96m_N_Spd_1Hz_Raw_Max',
                    'Aneo_96m_N_Spd_1Hz_Raw_Min', 'Aneo_96m_S_Spd_1Hz_Raw_Avg', 'Aneo_96m_S_Spd_1Hz_Raw_Std',
                    'Aneo_96m_S_Spd_1Hz_Raw_Max', 'Aneo_96m_S_Spd_1Hz_Raw_Min', 'Aneo_92m_NW_Spd_1Hz_Raw_Avg',
                    'Aneo_92m_NW_Spd_1Hz_Raw_Std', 'Aneo_92m_NW_Spd_1Hz_Raw_Max', 'Aneo_92m_NW_Spd_1Hz_Raw_Min',
                    'Aneo_70m_NW_Spd_1Hz_Raw_Avg', 'Aneo_70m_NW_Spd_1Hz_Raw_Std', 'Aneo_70m_NW_Spd_1Hz_Raw_Max',
                    'Aneo_70m_NW_Spd_1Hz_Raw_Min', 'Aneo_48m_NW_Spd_1Hz_Raw_Avg', 'Aneo_48m_NW_Spd_1Hz_Raw_Std',
                    'Aneo_48m_NW_Spd_1Hz_Raw_Max', 'Aneo_48m_NW_Spd_1Hz_Raw_Min', 'Aneo_32m_NW_Spd_1Hz_Raw_Avg',
                    'Aneo_32m_NW_Spd_1Hz_Raw_Std', 'Aneo_32m_NW_Spd_1Hz_Raw_Max', 'Aneo_32m_NW_Spd_1Hz_Raw_Min',
                    'Vane_87m_NW_Dir_1Hz_Raw_Avg', 'Vane_87m_NW_Dir_1Hz_Raw_Std', 'Vane_87m_NW_Dir_1Hz_Raw_Max',
                    'Vane_87m_NW_Dir_1Hz_Raw_Min', 'Vane_32m_NW_Dir_1Hz_Raw_Avg', 'Vane_32m_NW_Dir_1Hz_Raw_Std',
                    'Vane_32m_NW_Dir_1Hz_Raw_Max', 'Vane_32m_NW_Dir_1Hz_Raw_Min', 'GeoV_87m_NW_Dir_1Hz_Raw_Avg',
                    'GeoV_87m_NW_Dir_1Hz_Raw_Std', 'GeoV_87m_NW_Dir_1Hz_Raw_Max', 'GeoV_87m_NW_Dir_1Hz_Raw_Min',
                    'GeoV_87m_NW_Temp_1Hz_Raw_Avg', 'GeoV_87m_NW_Temp_1Hz_Raw_Std', 'GeoV_87m_NW_Temp_1Hz_Raw_Max',
                    'GeoV_87m_NW_Temp_1Hz_Raw_Min', 'GeoV_87m_NW_Status_1Hz_Raw', 'USA3D_92m_S_u_1Hz_Raw_Avg',
                    'USA3D_92m_S_u_1Hz_Raw_Std', 'USA3D_92m_S_u_1Hz_Raw_Max', 'USA3D_92m_S_u_1Hz_Raw_Min',
                    'USA3D_92m_S_v_1Hz_Raw_Avg', 'USA3D_92m_S_v_1Hz_Raw_Std', 'USA3D_92m_S_v_1Hz_Raw_Max',
                    'USA3D_92m_S_v_1Hz_Raw_Min', 'USA3D_92m_S_w_1Hz_Raw_Avg', 'USA3D_92m_S_w_1Hz_Raw_Std',
                    'USA3D_92m_S_w_1Hz_Raw_Max', 'USA3D_92m_S_w_1Hz_Raw_Min', 'USA3D_92m_S_Spd_1Hz_Raw_Avg',
                    'USA3D_92m_S_Spd_1Hz_Raw_Std', 'USA3D_92m_S_Spd_1Hz_Raw_Max', 'USA3D_92m_S_Spd_1Hz_Raw_Min',
                    'USA3D_92m_S_Dir_1Hz_Raw_Avg', 'USA3D_92m_S_Dir_1Hz_Raw_Std', 'USA3D_92m_S_Dir_1Hz_Raw_Max',
                    'USA3D_92m_S_Dir_1Hz_Raw_Min', 'USA3D_92m_S_Inc_1Hz_Raw_Avg', 'USA3D_92m_S_Inc_1Hz_Raw_Std',
                    'USA3D_92m_S_Inc_1Hz_Raw_Max', 'USA3D_92m_S_Inc_1Hz_Raw_Min', 'USA3D_92m_S_FloTC_1Hz_Raw_Avg',
                    'USA3D_92m_S_FloTC_1Hz_Raw_Std', 'USA3D_92m_S_FloTC_1Hz_Raw_Max', 'USA3D_92m_S_FloTC_1Hz_Raw_Min',
                    'USA3D_32m_S_u_1Hz_Raw_Avg', 'USA3D_32m_S_u_1Hz_Raw_Std', 'USA3D_32m_S_u_1Hz_Raw_Max',
                    'USA3D_32m_S_u_1Hz_Raw_Min', 'USA3D_32m_S_v_1Hz_Raw_Avg', 'USA3D_32m_S_v_1Hz_Raw_Std',
                    'USA3D_32m_S_v_1Hz_Raw_Max', 'USA3D_32m_S_v_1Hz_Raw_Min', 'USA3D_32m_S_w_1Hz_Raw_Avg',
                    'USA3D_32m_S_w_1Hz_Raw_Std', 'USA3D_32m_S_w_1Hz_Raw_Max', 'USA3D_32m_S_w_1Hz_Raw_Min',
                    'USA3D_32m_S_Spd_1Hz_Raw_Avg', 'USA3D_32m_S_Spd_1Hz_Raw_Std', 'USA3D_32m_S_Spd_1Hz_Raw_Max',
                    'USA3D_32m_S_Spd_1Hz_Raw_Min', 'USA3D_32m_S_Dir_1Hz_Raw_Avg', 'USA3D_32m_S_Dir_1Hz_Raw_Std',
                    'USA3D_32m_S_Dir_1Hz_Raw_Max', 'USA3D_32m_S_Dir_1Hz_Raw_Min', 'USA3D_32m_S_Inc_1Hz_Raw_Avg',
                    'USA3D_32m_S_Inc_1Hz_Raw_Std', 'USA3D_32m_S_Inc_1Hz_Raw_Max', 'USA3D_32m_S_Inc_1Hz_Raw_Min',
                    'USA3D_32m_S_FloTC_1Hz_Raw_Avg', 'USA3D_32m_S_FloTC_1Hz_Raw_Std', 'USA3D_32m_S_FloTC_1Hz_Raw_Max',
                    'USA3D_32m_S_FloTC_1Hz_Raw_Min', 'Temp_96m_TC_1Hz_Raw_Avg', 'Temp_96m_TC_1Hz_Raw_Std',
                    'Temp_96m_TC_1Hz_Raw_Max', 'Temp_96m_TC_1Hz_Raw_Min', 'Temp_CR_TC_1Hz_Raw_Avg',
                    'Temp_CR_TC_1Hz_Raw_Std',
                    'Temp_CR_TC_1Hz_Raw_Max', 'Temp_CR_TC_1Hz_Raw_Min', 'RHS_96m_RH_1Hz_Raw_Avg',
                    'RHS_96m_RH_1Hz_Raw_Std',
                    'RHS_96m_RH_1Hz_Raw_Max', 'RHS_96m_RH_1Hz_Raw_Min', 'RHS_CR_RH_1Hz_Raw_Avg',
                    'RHS_CR_RH_1Hz_Raw_Std',
                    'RHS_CR_RH_1Hz_Raw_Max', 'RHS_CR_RH_1Hz_Raw_Min', 'Pres_96m_PR_1Hz_Raw_Avg',
                    'Pres_96m_PR_1Hz_Raw_Std',
                    'Pres_96m_PR_1Hz_Raw_Max', 'Pres_96m_PR_1Hz_Raw_Min', 'Pres_C_PR_1Hz_Raw_Avg',
                    'Pres_C_PR_1Hz_Raw_Std',
                    'Pres_C_PR_1Hz_Raw_Max', 'Pres_C_PR_1Hz_Raw_Min', 'TempD_CR_96m_dTC_1Hz_Raw_Avg',
                    'TempD_CR_96m_dTC_1Hz_Raw_Std', 'TempD_CR_96m_dTC_1Hz_Raw_Max', 'TempD_CR_96m_dTC_1Hz_Raw_Min',
                    'SST01_Sky_MD_1Hz_Raw_Avg', 'SST01_Sky_MD_1Hz_Raw_Std', 'SST01_Sky_MD_1Hz_Raw_Max',
                    'SST01_Sky_MD_1Hz_Raw_Min', 'SST01_Sea_MD_1Hz_Raw_Avg', 'SST01_Sea_MD_1Hz_Raw_Std',
                    'SST01_Sea_MD_1Hz_Raw_Max', 'SST01_Sea_MD_1Hz_Raw_Min', 'DL_IntTemp_Max', 'DL_IntTemp_Min',
                    'DL_IntTemp_Std',
                    'SupplyVolt_Max', 'SupplyVolt_Min', 'SupplyVolt_Std']

    for c in del_channels:
        del df[c]

    # Join pre-processed 8Hz -> 1Hz downsampled data
    # Logic to find and load corresponding periods of 8Hz data
    start = df.first_valid_index()
    end = df.last_valid_index()
    files_1Hz = sorted(path_1Hz.glob('*.nc'))


    # Build list of 1Hz files to load based on timestamp range in source file (10min)
    timestamp = []
    for f in files_1Hz:
        timestamp.append(str(f).split('1Hz_')[-1].split('.nc')[0])

    df_1Hz_files = pd.DataFrame(timestamp, columns=['timestamp'])

    df_1Hz_files[['start', 'end']] = df_1Hz_files['timestamp'].str.split('_', expand=True)
    df_1Hz_files['start'] = pd.to_datetime(df_1Hz_files['start'], format='%Y-%m-%d-%H-%M-%S', errors='coerce')
    df_1Hz_files['end'] = pd.to_datetime(df_1Hz_files['end'], format='%Y-%m-%d-%H-%M-%S', errors='coerce')
    df_1Hz_files['filename'] = files_1Hz
    del df_1Hz_files['timestamp']

    # Find corresponding files that match current period
    indices = df_1Hz_files[(df_1Hz_files['start'] >= start - pd.Timedelta('10Min')) &
                           (df_1Hz_files['end'] <= end + pd.Timedelta('10Min'))].index
    indices = indices.insert(0, indices[0] - 1)
    indices = indices.insert(len(indices), indices[-1] + 1)
    # print(indices)

    # Ensure we don't go out of bounds at the end
    if any(i > df_1Hz_files.index[-1] for i in indices):
        indices = indices[0:-1]
        print('Reached end of 1Hz files')

    # Make list of files to load
    files_1Hz_load = list(df_1Hz_files.iloc[indices]['filename'].values)

    # Load the file list, resample and join channels of interest
    # Avoid value error on monotonic global indexes
    ds_1Hz = xr.open_mfdataset(files_1Hz_load, chunks='auto')

    df_1Hz_raw = ds_1Hz.to_pandas()
    del df_1Hz_raw['RECORD']
    
    # Apply wind vane averaging before resampling (to avoid 0/360 boundary)
    df_vane_10min = vane_dir_correction(df_1Hz_raw)
    
    # Get the mean, min, max, and std of 10-minute averages
    df_1Hz_mean = df_1Hz_raw.resample('10Min', label='right').mean()
    
    # Append Avg tag to channel names of downsampled data
    new_columns_avg = {column: column + "_Avg" for column in df_1Hz_raw.columns}
    df_1Hz_mean.rename(columns=new_columns_avg, inplace=True)
    
    # Apply sonic reconstructions
    df_1Hz_mean = sonic_reconstruction(df_1Hz_mean)
    
    # Join vane pre-calculated result to other averaged channels
    df_1Hz_mean['Vane_32m_NW_Dir_1Hz_Raw_Avg'] = df_vane_10min
    
    # Continue with resample operation after direction corrections
    df_1Hz_min = df_1Hz_raw.resample('10Min', label='right').min()
    df_1Hz_max = df_1Hz_raw.resample('10Min', label='right').max()
    df_1Hz_std = df_1Hz_raw.resample('10Min', label='right').std()

    # Append Min tag to channel names of downsampled data
    new_columns_min = {column: column + "_Avg_Min" for column in df_1Hz_raw.columns}
    df_1Hz_min.rename(columns=new_columns_min, inplace=True)

    # Do the same for the maximums
    new_columns_max = {column: column + "_Avg_Max" for column in df_1Hz_raw.columns}
    df_1Hz_max.rename(columns=new_columns_max, inplace=True)

    # Do the same for the standard deviations
    new_columns_std = {column: column + "_Avg_Std" for column in df_1Hz_raw.columns}
    df_1Hz_std.rename(columns=new_columns_std, inplace=True)

    # Column names which have already been downsampled 8Hz->1Hz already have _Avg appended. Rename those so they aren't duplicated.
    for col in df_1Hz_mean.columns:
        if (col.split('_')[-2] == 'Avg'):
            new_col = '_'.join(col.split('_')[0:-1])
            df_1Hz_mean.rename(columns={col: new_col}, inplace=True)

    for col in df_1Hz_min.columns:
        if (col.split('_')[-3] == 'Avg'):
            new_col = '_'.join(col.split('_')[0:-2]) + '_' + col.split('_')[-1]
            df_1Hz_min.rename(columns={col: new_col}, inplace=True)

    for col in df_1Hz_max.columns:
        if (col.split('_')[-3] == 'Avg'):
            new_col = '_'.join(col.split('_')[0:-2]) + '_' + col.split('_')[-1]
            df_1Hz_max.rename(columns={col: new_col}, inplace=True)

    for col in df_1Hz_std.columns:
        if (col.split('_')[-3] == 'Avg'):
            new_col = '_'.join(col.split('_')[0:-2]) + '_' + col.split('_')[-1]
            df_1Hz_std.rename(columns={col: new_col}, inplace=True)

    # Join all the downsampled 1Hz dataframes to the source 10-min data
    df = df.join(df_1Hz_mean, how='left')
    df = df.join(df_1Hz_min, how='left')
    df = df.join(df_1Hz_max, how='left')
    df = df.join(df_1Hz_std, how='left')

    # Join Fraunhofer data provided by Pedro Santos
    # This is incredibly inefficient because it loads the entire IWES dataset for every 1-day file from the data logger
    # 92m data
    df_iwes_92m = pd.read_csv(path_iwes_92m)
    df_iwes_92m.set_index(pd.to_datetime(df_iwes_92m['Time']), inplace=True)
    del df_iwes_92m['Time']

    # Some column names have whitespace at end
    df_iwes_92m.columns = df_iwes_92m.columns.str.replace(' ', '')

    # Remove the units tag at the end of the column names in header
    for col in df_iwes_92m.columns:
        new_col = col.split('[')[0]
        df_iwes_92m.rename(columns={col: new_col}, inplace=True)

    # Append IWES_92m to appropriate channel names
    new_columns_iwes_92m = {column: column + "_IWES_92m" for column in df_iwes_92m.columns}
    df_iwes_92m.rename(columns=new_columns_iwes_92m, inplace=True)

    # 32m data
    df_iwes_32m = pd.read_csv(path_iwes_32m)
    df_iwes_32m.set_index(pd.to_datetime(df_iwes_32m['Time']), inplace=True)
    del df_iwes_32m['Time']

    # Some column names have whitespace at end
    df_iwes_32m.columns = df_iwes_32m.columns.str.replace(' ', '')

    # Remove the units tag at the end of the column names in header
    for col in df_iwes_32m.columns:
        new_col = col.split('[')[0]
        df_iwes_32m.rename(columns={col: new_col}, inplace=True)

    # Append IWES_32m to appropriate channel names
    new_columns_iwes_32m = {column: column + "_IWES_32m" for column in df_iwes_32m.columns}
    df_iwes_32m.rename(columns=new_columns_iwes_32m, inplace=True)

    # Join the IWES data to the met-mast data logger channels
    # Using these now for validation- should we keep them all in the final dataset?
    df = df.join(df_iwes_92m, how='left')
    df = df.join(df_iwes_32m, how='left')

    # Convert pandas._libs.missing.NAType to np.nan
    df.fillna(np.nan, inplace=True)

    return df

def vane_dir_correction(df_1Hz_raw):

    # Correct wind vane averaging
    # Offset is already applied in 1Hz dataset
    vane_rads = np.deg2rad(df_1Hz_raw['Vane_32m_NW_Dir_1Hz_Raw'])
    df_vane_10min = (np.rad2deg(np.arctan2((np.sin(vane_rads).resample('10Min').mean()), (np.cos(vane_rads).resample('10Min').mean())))+360)%360
    
    return df_vane_10min

def sonic_reconstruction(df_1Hz_mean):
    
    # Repeat sonic reconstruction following resampling (otherwise direction averaging near 0 degrees is incorrect)
    del df_1Hz_mean['USA3D_92m_S_Spd_8Hz_Calc_Avg_Avg']
    del df_1Hz_mean['USA3D_32m_S_Spd_8Hz_Calc_Avg_Avg']
    del df_1Hz_mean['USA3D_92m_S_Dir_8Hz_Calc_Avg_Avg']
    del df_1Hz_mean['USA3D_32m_S_Dir_8Hz_Calc_Avg_Avg']

    # Re-do reconstruction of sonic speed and direction
    # 3D reconstuction (x,y,z)

    df_1Hz_mean['USA3D_92m_S_Spd_8Hz_Calc_Avg_Avg'] = np.sqrt(df_1Hz_mean['USA3D_92m_S_x_8Hz_Raw_Avg_Avg']**2 + df_1Hz_mean['USA3D_92m_S_y_8Hz_Raw_Avg_Avg']**2 + df_1Hz_mean['USA3D_92m_S_z_8Hz_Raw_Avg_Avg']**2)
    df_1Hz_mean['USA3D_32m_S_Spd_8Hz_Calc_Avg_Avg'] = np.sqrt(df_1Hz_mean['USA3D_32m_S_x_8Hz_Raw_Avg_Avg']**2 + df_1Hz_mean['USA3D_32m_S_y_8Hz_Raw_Avg_Avg']**2 + df_1Hz_mean['USA3D_32m_S_z_8Hz_Raw_Avg_Avg']**2)

    df_1Hz_mean['USA3D_92m_S_Dir_8Hz_Calc_Avg_Avg'] = (np.rad2deg(np.arctan2(df_1Hz_mean['USA3D_92m_S_y_8Hz_Raw_Avg_Avg'], df_1Hz_mean['USA3D_92m_S_x_8Hz_Raw_Avg_Avg']))+360)%360
    df_1Hz_mean['USA3D_32m_S_Dir_8Hz_Calc_Avg_Avg'] = (np.rad2deg(np.arctan2(df_1Hz_mean['USA3D_32m_S_y_8Hz_Raw_Avg_Avg'], df_1Hz_mean['USA3D_32m_S_x_8Hz_Raw_Avg_Avg']))+360)%360

    # Apply offset to sonic wind directions

    sonic_32m_dir_offset = 208.52
    sonic_92m_dir_offset = 211.66

    df_1Hz_mean['USA3D_32m_S_Dir_8Hz_Calc_Avg_Avg'] = (df_1Hz_mean['USA3D_32m_S_Dir_8Hz_Calc_Avg_Avg']+sonic_32m_dir_offset+360)%360
    df_1Hz_mean['USA3D_92m_S_Dir_8Hz_Calc_Avg_Avg'] = (df_1Hz_mean['USA3D_92m_S_Dir_8Hz_Calc_Avg_Avg']+sonic_92m_dir_offset+360)%360

    return df_1Hz_mean

def format_xds(df):
    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(df)

    # Ensure missing data is correct format for netcdf file (must be np.nan (float object) and not pandas.NaT)
    ds = ds.fillna(np.nan)

    # Cast xarray variables into correct dtypes (unsure why not preserved when going pandas -> xarray)
    for c in df.columns:
        if c in ['RECORD', 'NumSamples', 'SkppdRecord1', 'SkppdRecord2', 'NTP_PingServerResp', 'NTP_TimeOffset',
                 'SkppdScan', 'SkppdSlwScan1', 'SkppdSlwScan2', 'SkppdSlwScan3', 'SkppdSlwScan4']:
            ds[c] = ds[c].astype(np.int64)
        elif c == 'NTP_DtTmSync':
            ds[c] = ds[c].astype(str)
        else:
            ds[c] = ds[c].astype(np.float64)

    # Ensure TIMESTAMP is dt64 and not pandas timestamp etc.
    ds['TIMESTAMP'] = ds['TIMESTAMP'].astype(np.datetime64)

    return ds


def add_metadata(ds):

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

    ds['Temp_96m_TC_1Hz_Raw_Avg'].attrs['description'] = 'Air temperature at 96m height. Downsampled to 10-minute average'
    ds['Temp_96m_TC_1Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_96m_TC_1Hz_Raw_Avg'].attrs['height'] = '96m'

    ds['Temp_CR_TC_1Hz_Raw_Avg'].attrs['description'] = 'Air temperature at container roof position, 24.66m height. Downsampled to 10-minute average'
    ds['Temp_CR_TC_1Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_CR_TC_1Hz_Raw_Avg'].attrs['height'] = '24.66m'

    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg'].attrs['description'] = 'Air temperature difference between 96m and container roof (24.66m). Downsampled to 10-minute average'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg'].attrs['height'] = '96m'

    ds['RHS_96m_RH_1Hz_Raw_Avg'].attrs['description'] = 'Relative humidity at 96m height. Downsampled to 10-minute average'
    ds['RHS_96m_RH_1Hz_Raw_Avg'].attrs['units'] = 'Percent'
    ds['RHS_96m_RH_1Hz_Raw_Avg'].attrs['height'] = '96m'

    ds['RHS_CR_RH_1Hz_Raw_Avg'].attrs['description'] = 'Relative humidity at container roof position, 24.66m height. Downsampled to 10-minute average'
    ds['RHS_CR_RH_1Hz_Raw_Avg'].attrs['units'] = 'Percent'
    ds['RHS_CR_RH_1Hz_Raw_Avg'].attrs['height'] = '24.66m'

    ds['Pres_96m_PR_1Hz_Raw_Avg'].attrs['description'] = 'Air pressure at 96m height. Downsampled to 10-minute average'
    ds['Pres_96m_PR_1Hz_Raw_Avg'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_96m_PR_1Hz_Raw_Avg'].attrs['height'] = '96m'

    ds['Pres_C_PR_1Hz_Raw_Avg'].attrs['description'] = 'Air pressure at container position, 22.3m height. Downsampled to 10-minute average'
    ds['Pres_C_PR_1Hz_Raw_Avg'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_C_PR_1Hz_Raw_Avg'].attrs['height'] = '22.3m'

    ds['SST01_Sky_MD_1Hz_Raw_Avg'].attrs['description'] = 'Infrared radiometer reading, pointed up at the sky. Downsampled to 10-minute average'
    ds['SST01_Sky_MD_1Hz_Raw_Avg'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sky_MD_1Hz_Raw_Avg'].attrs['height'] = '20m'

    ds['SST01_Sea_MD_1Hz_Raw_Avg'].attrs['description'] = 'Infrared radiometer reading, pointed down at the sea surface. Downsampled to 10-minute average'
    ds['SST01_Sea_MD_1Hz_Raw_Avg'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sea_MD_1Hz_Raw_Avg'].attrs['height'] = '20m'

    ds['SST01_Sea_1Hz_RadEm_Calc_Avg'].attrs['description'] = 'Radiant emission calculated from the infrared radiometer pointed down at sea surface using a calculated spectral radiance curve applied using a lookup table. Downsampled to 10-minute average'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg'].attrs['units'] = 'Watt/steradian/square meter'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg'].attrs['height'] = '20m'

    ds['SST_SkyCorrection_1Hz_Calc_Avg'].attrs['description'] = 'Sea surface temperature with sky correction method applied to correct for background radiation. Downsampled to 10-minute average'
    ds['SST_SkyCorrection_1Hz_Calc_Avg'].attrs['units'] = 'Degrees Kelvin'
    ds['SST_SkyCorrection_1Hz_Calc_Avg'].attrs['height'] = '20m'

    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg'].attrs['description'] = 'Wind vane (North-West boom) wind direction at 32m height with offset applied. Downsampled to 10-minute average'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg'].attrs['units'] = 'Degrees'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg'].attrs['description'] = 'Cup anemometer (North boom) wind speed at 96m height with calibration gains/offsets applied. Downsampled to 10-minute average'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg'].attrs['height'] = '96m'

    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg'].attrs['description'] = 'Cup anemometer (South boom) wind speed at 96m height with calibration gains/offsets applied. Downsampled to 10-minute average'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg'].attrs['height'] = '96m'

    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg'].attrs['description'] = 'Cup anemometer (North-West boom) wind speed at 92m height with calibration gains/offsets applied. Downsampled to 10-minute average'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg'].attrs['description'] = 'Cup anemometer (North-West boom) wind speed at 70m height with calibration gains/offsets applied. Downsampled to 10-minute average'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg'].attrs['height'] = '70m'

    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg'].attrs['description'] = 'Cup anemometer (North-West boom) wind speed at 48m height with calibration gains/offsets applied. Downsampled to 10-minute average'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg'].attrs['height'] = '48m'

    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg'].attrs['description'] = 'Cup anemometer (North-West boom) wind speed at 32m height with calibration gains/offsets applied. Downsampled to 10-minute average'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_x_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) x-axis speed at 92m height. Outliers removed using moving-median filter. Downsampled to 10-minute average'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_92m_S_y_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) y-axis speed at 92m height. Outliers removed using moving-median filter. Downsampled to 10-minute average'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_92m_S_z_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) z-axis speed at 92m height. Outliers removed using moving-median filter. Downsampled to 10-minute average'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_x_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) x-axis speed at 32m height. Outliers removed using moving-median filter. Downsampled to 10-minute average'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_32m_S_y_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) y-axis speed at 32m height. Outliers removed using moving-median filter. Downsampled to 10-minute average'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_32m_S_z_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) z-axis speed at 32m height. Outliers removed using moving-median filter. Downsampled to 10-minute average'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) temperature at 92m height. Downsampled to 10-minute average'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) temperature at 32m height. Downsampled to 10-minute average'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) flow inclination at 92m height, raw output. Downsampled to 1Hz'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg'].attrs['description'] = 'Sonic anemometer (South boom) flow inclination at 32m height, raw output. Downsampled to 1Hz'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg'].attrs['description'] = 'Sonic anemometer (South boom) flow inclination at 92m height calculated using: arctan2(z,sqrt(x^2+y^2)). Downsampled to 10-minute average'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg'].attrs['description'] = 'Sonic anemometer (South boom) flow inclination at 32m height calculated using: arctan2(z,sqrt(x^2+y^2)). Downsampled to 10-minute average'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg'].attrs['description'] = 'Sonic anemometer (South boom) wind speed at 92m height calculated using 3D reconstruction: sqrt(x^2+y^2+z^2). Downsampled to 10-minute average'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg'].attrs['description'] = 'Sonic anemometer (South boom) wind speed at 32m height calculated using 3D reconstruction: sqrt(x^2+y^2+z^2). Downsampled to 10-minute average'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg'].attrs['description'] = 'Sonic anemometer (South boom) wind direction at 92m height calculated using 2D reconstruction: (rad2deg(arctan2(y,x))+360)%360 with offset applied. Downsampled to 10-minute average'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg'].attrs['description'] = 'Sonic anemometer (South boom) wind direction at 32m height calculated using 2D reconstruction: (rad2deg(arctan2(y,x))+360)%360 with offset applied. Downsampled to 10-minute average'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg'].attrs['height'] = '32m'

    ds['NumSamples'].attrs['description'] = 'Number of samples within 10-minute average'
    ds['NumSamples'].attrs['units'] = 'Integer'
    ds['NumSamples'].attrs['height'] = 'N/A'

    ds['DL_IntTemp_Avg'].attrs['description'] = 'Data logger internal temperature'
    ds['DL_IntTemp_Avg'].attrs['units'] = 'Degrees Celsius'
    ds['DL_IntTemp_Avg'].attrs['height'] = 'N/A'

    ds['SupplyVolt_Avg'].attrs['description'] = 'Power supply voltage'
    ds['SupplyVolt_Avg'].attrs['units'] = 'Volts'
    ds['SupplyVolt_Avg'].attrs['height'] = 'N/A'

    ds['SkppdRecord1'].attrs['description'] = 'Indicates when a record was not stored to the data logger when it should have been'
    ds['SkppdRecord1'].attrs['units'] = 'Integer'
    ds['SkppdRecord1'].attrs['height'] = 'N/A'

    ds['SkppdRecord2'].attrs['description'] = 'Indicates when a record was not stored to the data logger when it should have been'
    ds['SkppdRecord2'].attrs['units'] = 'Integer'
    ds['SkppdRecord2'].attrs['height'] = 'N/A'

    ds['NTP_PingServerResp'].attrs['description'] = 'Response time to network time server'
    ds['NTP_PingServerResp'].attrs['units'] = 'Milliseconds'
    ds['NTP_PingServerResp'].attrs['height'] = 'N/A'

    ds['NTP_TimeOffset'].attrs['description'] = 'Time offset between data logger clock and NTP time reference'
    ds['NTP_TimeOffset'].attrs['units'] = 'Milliseconds'
    ds['NTP_TimeOffset'].attrs['height'] = 'N/A'

    ds['SkppdScan'].attrs['description'] = 'Indicates when sampling on the data logger took longer to process than the scan interval allowed leading to potentially missing data'
    ds['SkppdScan'].attrs['units'] = 'Integer'
    ds['SkppdScan'].attrs['height'] = 'N/A'

    ds['SkppdSlwScan1'].attrs['description'] = 'Indicates when sampling on the data logger took longer to process than the scan interval allowed leading to potentially missing data'
    ds['SkppdSlwScan1'].attrs['units'] = 'Integer'
    ds['SkppdSlwScan1'].attrs['height'] = 'N/A'

    ds['SkppdSlwScan2'].attrs['description'] = 'Indicates when sampling on the data logger took longer to process than the scan interval allowed leading to potentially missing data'
    ds['SkppdSlwScan2'].attrs['units'] = 'Integer'
    ds['SkppdSlwScan2'].attrs['height'] = 'N/A'

    ds['SkppdSlwScan3'].attrs['description'] = 'Indicates when sampling on the data logger took longer to process than the scan interval allowed leading to potentially missing data'
    ds['SkppdSlwScan3'].attrs['units'] = 'Integer'
    ds['SkppdSlwScan3'].attrs['height'] = 'N/A'

    ds['SkppdSlwScan4'].attrs['description'] = 'Indicates when sampling on the data logger took longer to process than the scan interval allowed leading to potentially missing data'
    ds['SkppdSlwScan4'].attrs['units'] = 'Integer'
    ds['SkppdSlwScan4'].attrs['height'] = 'N/A'

    ds['NTP_DtTmSync'].attrs['description'] = 'Timestamp of last NTP time synchonization'
    ds['NTP_DtTmSync'].attrs['units'] = 'String DateTime timestamp'
    ds['NTP_DtTmSync'].attrs['height'] = 'N/A'

    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Min'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Min'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Min'].attrs['height'] = '92m'

    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Std'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Std'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Std'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_Spd_8Hz_Calc_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Max'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Calc_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Max'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Dir_8Hz_Calc_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_x_8Hz_Raw_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_y_8Hz_Raw_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_z_8Hz_Raw_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_92m_S_FloTC_8Hz_Raw_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Raw_Avg_Max'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Min'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Min'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Min'].attrs['height'] = '32m'

    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Std'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Std'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Std'].attrs['height'] = '32m'

    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_x_8Hz_Raw_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_y_8Hz_Raw_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_z_8Hz_Raw_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_32m_S_FloTC_8Hz_Raw_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Raw_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_Spd_8Hz_Calc_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Max'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Dir_8Hz_Calc_Avg_Max'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Max'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Calc_Avg_Max'].attrs['height'] = '32m'

    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Max'].attrs['height'] = '32m'

    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Std'].attrs['height'] = '32m'

    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees'
    ds['Vane_32m_NW_Dir_1Hz_Raw_Avg_Min'].attrs['height'] = '32m'

    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Min'].attrs['height'] = '96m'

    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Min'].attrs['height'] = '96m'

    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['height'] = '92m'

    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['height'] = '70m'

    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['height'] = '48m'

    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['units'] = 'Meters/second'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Min'].attrs['height'] = '32m'

    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Max'].attrs['height'] = '96m'

    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Max'].attrs['height'] = '96m'

    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['height'] = '92m'

    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['height'] = '70m'

    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['height'] = '48m'

    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['units'] = 'Meters/second'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Max'].attrs['height'] = '32m'

    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_N_Spd_1Hz_Calc_Avg_Std'].attrs['height'] = '96m'

    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['Aneo_96m_S_Spd_1Hz_Calc_Avg_Std'].attrs['height'] = '96m'

    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['Aneo_92m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['height'] = '92m'

    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['Aneo_70m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['height'] = '70m'

    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['Aneo_48m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['height'] = '48m'

    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['units'] = 'Meters/second'
    ds['Aneo_32m_NW_Spd_1Hz_Calc_Avg_Std'].attrs['height'] = '32m'

    ds['Temp_96m_TC_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Temp_96m_TC_1Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_96m_TC_1Hz_Raw_Avg_Min'].attrs['height'] = '96m'

    ds['Temp_CR_TC_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Temp_CR_TC_1Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_CR_TC_1Hz_Raw_Avg_Min'].attrs['height'] = '24.66m'

    ds['Temp_96m_TC_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Temp_96m_TC_1Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_96m_TC_1Hz_Raw_Avg_Max'].attrs['height'] = '96m'

    ds['Temp_CR_TC_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Temp_CR_TC_1Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_CR_TC_1Hz_Raw_Avg_Max'].attrs['height'] = '24.66m'

    ds['Temp_96m_TC_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Temp_96m_TC_1Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_96m_TC_1Hz_Raw_Avg_Std'].attrs['height'] = '96m'

    ds['Temp_CR_TC_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Temp_CR_TC_1Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Celsius'
    ds['Temp_CR_TC_1Hz_Raw_Avg_Std'].attrs['height'] = '24.66m'

    ds['RHS_96m_RH_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['RHS_96m_RH_1Hz_Raw_Avg_Std'].attrs['units'] = 'Percent'
    ds['RHS_96m_RH_1Hz_Raw_Avg_Std'].attrs['height'] = '96m'

    ds['RHS_CR_RH_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['RHS_CR_RH_1Hz_Raw_Avg_Std'].attrs['units'] = 'Percent'
    ds['RHS_CR_RH_1Hz_Raw_Avg_Std'].attrs['height'] = '24.66m'

    ds['RHS_96m_RH_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['RHS_96m_RH_1Hz_Raw_Avg_Min'].attrs['units'] = 'Percent'
    ds['RHS_96m_RH_1Hz_Raw_Avg_Min'].attrs['height'] = '96m'

    ds['RHS_CR_RH_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['RHS_CR_RH_1Hz_Raw_Avg_Min'].attrs['units'] = 'Percent'
    ds['RHS_CR_RH_1Hz_Raw_Avg_Min'].attrs['height'] = '24.66m'

    ds['RHS_96m_RH_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['RHS_96m_RH_1Hz_Raw_Avg_Max'].attrs['units'] = 'Percent'
    ds['RHS_96m_RH_1Hz_Raw_Avg_Max'].attrs['height'] = '96m'

    ds['RHS_CR_RH_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['RHS_CR_RH_1Hz_Raw_Avg_Max'].attrs['units'] = 'Percent'
    ds['RHS_CR_RH_1Hz_Raw_Avg_Max'].attrs['height'] = '24.66m'

    ds['Pres_96m_PR_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Pres_96m_PR_1Hz_Raw_Avg_Min'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_96m_PR_1Hz_Raw_Avg_Min'].attrs['height'] = '96m'

    ds['Pres_C_PR_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['Pres_C_PR_1Hz_Raw_Avg_Min'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_C_PR_1Hz_Raw_Avg_Min'].attrs['height'] = '22.3m'

    ds['Pres_96m_PR_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Pres_96m_PR_1Hz_Raw_Avg_Std'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_96m_PR_1Hz_Raw_Avg_Std'].attrs['height'] = '96m'

    ds['Pres_C_PR_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['Pres_C_PR_1Hz_Raw_Avg_Std'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_C_PR_1Hz_Raw_Avg_Std'].attrs['height'] = '22.3m'

    ds['Pres_96m_PR_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Pres_96m_PR_1Hz_Raw_Avg_Max'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_96m_PR_1Hz_Raw_Avg_Max'].attrs['height'] = '96m'

    ds['Pres_C_PR_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['Pres_C_PR_1Hz_Raw_Avg_Max'].attrs['units'] = 'Millibar/Hectopascal'
    ds['Pres_C_PR_1Hz_Raw_Avg_Max'].attrs['height'] = '22.3m'

    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Celsius'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Min'].attrs['height'] = 'N/A'

    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Celsius'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Max'].attrs['height'] = 'N/A'

    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Celsius'
    ds['TempD_CR_96m_dTC_1Hz_Raw_Avg_Std'].attrs['height'] = 'N/A'

    ds['SST01_Sky_MD_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['SST01_Sky_MD_1Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sky_MD_1Hz_Raw_Avg_Max'].attrs['height'] = '20m'

    ds['SST01_Sky_MD_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['SST01_Sky_MD_1Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sky_MD_1Hz_Raw_Avg_Min'].attrs['height'] = '20m'

    ds['SST01_Sky_MD_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['SST01_Sky_MD_1Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sky_MD_1Hz_Raw_Avg_Std'].attrs['height'] = '20m'

    ds['SST01_Sea_MD_1Hz_Raw_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['SST01_Sea_MD_1Hz_Raw_Avg_Std'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sea_MD_1Hz_Raw_Avg_Std'].attrs['height'] = '20m'

    ds['SST01_Sea_MD_1Hz_Raw_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['SST01_Sea_MD_1Hz_Raw_Avg_Min'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sea_MD_1Hz_Raw_Avg_Min'].attrs['height'] = '20m'

    ds['SST01_Sea_MD_1Hz_Raw_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['SST01_Sea_MD_1Hz_Raw_Avg_Max'].attrs['units'] = 'Degrees Kelvin'
    ds['SST01_Sea_MD_1Hz_Raw_Avg_Max'].attrs['height'] = '20m'

    ds['SST_SkyCorrection_1Hz_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['SST_SkyCorrection_1Hz_Calc_Avg_Min'].attrs['units'] = 'Degrees Kelvin'
    ds['SST_SkyCorrection_1Hz_Calc_Avg_Min'].attrs['height'] = '20m'

    ds['SST_SkyCorrection_1Hz_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['SST_SkyCorrection_1Hz_Calc_Avg_Std'].attrs['units'] = 'Degrees Kelvin'
    ds['SST_SkyCorrection_1Hz_Calc_Avg_Std'].attrs['height'] = '20m'

    ds['SST_SkyCorrection_1Hz_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['SST_SkyCorrection_1Hz_Calc_Avg_Max'].attrs['units'] = 'Degrees Kelvin'
    ds['SST_SkyCorrection_1Hz_Calc_Avg_Max'].attrs['height'] = '20m'

    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Std'].attrs['description'] = 'Standard deviation within 10-minute period averaged'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Std'].attrs['units'] = 'Watt/steradian/square meter'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Std'].attrs['height'] = '20m'

    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Min'].attrs['description'] = 'Minimum within 10-minute period averaged'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Min'].attrs['units'] = 'Watt/steradian/square meter'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Min'].attrs['height'] = '20m'

    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Max'].attrs['description'] = 'Maximum within 10-minute period averaged'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Max'].attrs['units'] = 'Watt/steradian/square meter'
    ds['SST01_Sea_1Hz_RadEm_Calc_Avg_Max'].attrs['height'] = '20m'

    # Faunhofer

    ds['U_horz_IWES_92m'].attrs['description'] = '10min block average wind speed sampled over 20Hz (no detrending). The wind speed is the magnitude of the wind velocity. Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component.'
    ds['U_horz_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['U_horz_IWES_92m'].attrs['height'] = '92m'

    ds['U_vec_IWES_92m'].attrs['description'] = '10min block average wind speed sampled over 20Hz (no detrending). The wind speed is the magnitude of the 3D wind vector.'
    ds['U_vec_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['U_vec_IWES_92m'].attrs['height'] = '92m'

    ds['wind_direction_IWES_92m'].attrs['description'] = '10min block average wind direction sampled over 20Hz (no detrending)'
    ds['wind_direction_IWES_92m'].attrs['units'] = 'Degrees'
    ds['wind_direction_IWES_92m'].attrs['height'] = '92m'

    ds['inflow_angle_IWES_92m'].attrs['description'] = '10min block average wind direction sampled over 20Hz. Here, the direction of the wind vector is given as the direction from which it is blowing (wind_from_direction) (westerly, northerly, etc.)'
    ds['inflow_angle_IWES_92m'].attrs['units'] = 'Degrees'
    ds['inflow_angle_IWES_92m'].attrs['height'] = '92m'

    ds['u_IWES_92m'].attrs['description'] = '10min block average of the along-wind speed component. Also called stream-wise wind component. Is the wind component aligned with the main wind direction over a 10-min period by performing a pitch and roll sonic coordinate rotation'
    ds['u_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['u_IWES_92m'].attrs['height'] = '92m'

    ds['v_IWES_92m'].attrs['description'] = '10min block average of the cross-wind speed component. Is the wind component perpendicular to the main wind direction over a 10-min period after pitch and roll rotation'
    ds['v_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['v_IWES_92m'].attrs['height'] = '92m'

    ds['w_IWES_92m'].attrs['description'] = '10min block average of the vertical wind speed component after double rotation. Is the vertical wind component after pitch and roll sonic coordinate rotation in relation to the 10-min mean wind direction. Positive vertical wind is pointing upwards'
    ds['w_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['w_IWES_92m'].attrs['height'] = '92m'

    ds['T_IWES_92m'].attrs['description'] = '10min block average sonic temperature sampled at 20 Hz'
    ds['T_IWES_92m'].attrs['units'] = 'Degrees Kelvin'
    ds['T_IWES_92m'].attrs['height'] = '92m'

    ds['u_max_IWES_92m'].attrs['description'] = 'Maximum along-wind speed component (after double rotation) over 10-min period.'
    ds['u_max_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['u_max_IWES_92m'].attrs['height'] = '92m'

    ds['v_max_IWES_92m'].attrs['description'] = 'Maximum cross-wind component (after double rotation) over 10-min period.'
    ds['v_max_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['v_max_IWES_92m'].attrs['height'] = '92m'

    ds['w_max_IWES_92m'].attrs['description'] = 'Maximum vertical wind component (after double rotation) over 10-min period.'
    ds['w_max_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['w_max_IWES_92m'].attrs['height'] = '92m'

    ds['T_max_IWES_92m'].attrs['description'] = 'Maximum sonic temperature over 10-min period.'
    ds['T_max_IWES_92m'].attrs['units'] = 'Degrees Kelvin'
    ds['T_max_IWES_92m'].attrs['height'] = '92m'

    ds['u_min_IWES_92m'].attrs['description'] = 'Minimum along-wind speed component (after double rotation) over 10-min period.'
    ds['u_min_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['u_min_IWES_92m'].attrs['height'] = '92m'

    ds['v_min_IWES_92m'].attrs['description'] = 'Minimum cross-wind component (after double rotation) over 10-min period.'
    ds['v_min_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['v_min_IWES_92m'].attrs['height'] = '92m'

    ds['w_min_IWES_92m'].attrs['description'] = 'Minimum vertical wind component (after double rotation) over 10-min period.'
    ds['w_min_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['w_min_IWES_92m'].attrs['height'] = '92m'

    ds['T_min_IWES_92m'].attrs['description'] = 'Minimum sonic temperature over 10-min period.'
    ds['T_min_IWES_92m'].attrs['units'] = 'Degrees Kelvin'
    ds['T_min_IWES_92m'].attrs['height'] = '92m'

    ds['cov_uu_IWES_92m'].attrs['description'] = 'Auto-covariance of the along-wind speed component'
    ds['cov_uu_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uu_IWES_92m'].attrs['height'] = '92m'

    ds['cov_uv_IWES_92m'].attrs['description'] = 'Covariance of the along-wind and cross-wind speed components.'
    ds['cov_uv_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uv_IWES_92m'].attrs['height'] = '92m'

    ds['cov_uw_IWES_92m'].attrs['description'] = 'Covariance of the along-wind and vertical wind speed components.'
    ds['cov_uw_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uw_IWES_92m'].attrs['height'] = '92m'

    ds['cov_vv_IWES_92m'].attrs['description'] = 'Auto-covariance of the cross-wind speed component'
    ds['cov_vv_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_vv_IWES_92m'].attrs['height'] = '92m'

    ds['cov_vw_IWES_92m'].attrs['description'] = 'Covariance of the cross-wind and vertical wind speed components.'
    ds['cov_vw_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_vw_IWES_92m'].attrs['height'] = '92m'

    ds['cov_ww_IWES_92m'].attrs['description'] = 'Auto-covariance of the vertical wind speed component'
    ds['cov_ww_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_ww_IWES_92m'].attrs['height'] = '92m'

    ds['cov_uT_IWES_92m'].attrs['description'] = 'Covariance of the along-wind and temperature.'
    ds['cov_uT_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uT_IWES_92m'].attrs['height'] = '92m'

    ds['cov_vT_IWES_92m'].attrs['description'] = 'Covariance of the cross-wind and temperature.'
    ds['cov_vT_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_vT_IWES_92m'].attrs['height'] = '92m'

    ds['cov_wT_IWES_92m'].attrs['description'] = 'Upward sensible heat flux in the air using sonic temperature'
    ds['cov_wT_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_wT_IWES_92m'].attrs['height'] = '92m'

    ds['cov_TT_IWES_92m'].attrs['description'] = 'Auto-covariance of sonic temperature.'
    ds['cov_TT_IWES_92m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_TT_IWES_92m'].attrs['height'] = '92m'

    ds['U_horz_std_IWES_92m'].attrs['description'] = 'Standard deviation of the 10min horizontal wind speed sampled at 20Hz.'
    ds['U_horz_std_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['U_horz_std_IWES_92m'].attrs['height'] = '92m'

    ds['U_vec_std_IWES_92m'].attrs['description'] = 'Standard deviation of the 10min magnitude of the 3D wind vector sampled at 20Hz.'
    ds['U_vec_std_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['U_vec_std_IWES_92m'].attrs['height'] = '92m'

    ds['u_star_IWES_92m'].attrs['description'] = '10min block average Friction velocity sampled over 20Hz with along- and cross-wind components'
    ds['u_star_IWES_92m'].attrs['units'] = 'Meters/second'
    ds['u_star_IWES_92m'].attrs['height'] = '92m'

    ds['L-1_IWES_92m'].attrs['description'] = 'Inverse of the Obukhov Length'
    ds['L-1_IWES_92m'].attrs['units'] = 'Meters^-1'
    ds['L-1_IWES_92m'].attrs['height'] = '92m'

    ds['zL_IWES_92m'].attrs['description'] = 'Dimensionless stability parameter, where the Obukhov Length is scaled by the measurement sonic height'
    ds['zL_IWES_92m'].attrs['units'] = 'None'
    ds['zL_IWES_92m'].attrs['height'] = '92m'

    ds['U_horz_IWES_32m'].attrs['description'] = '10min block average wind speed sampled over 20Hz (no detrending). The wind speed is the magnitude of the wind velocity. Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component.'
    ds['U_horz_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['U_horz_IWES_32m'].attrs['height'] = '32m'

    ds['U_vec_IWES_32m'].attrs['description'] = '10min block average wind speed sampled over 20Hz (no detrending). The wind speed is the magnitude of the 3D wind vector.'
    ds['U_vec_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['U_vec_IWES_32m'].attrs['height'] = '32m'

    ds['wind_direction_IWES_32m'].attrs['description'] = '10min block average wind direction sampled over 20Hz (no detrending)'
    ds['wind_direction_IWES_32m'].attrs['units'] = 'Degrees'
    ds['wind_direction_IWES_32m'].attrs['height'] = '32m'

    ds['inflow_angle_IWES_32m'].attrs['description'] = '10min block average wind direction sampled over 20Hz. Here, the direction of the wind vector is given as the direction from which it is blowing (wind_from_direction) (westerly, northerly, etc.)'
    ds['inflow_angle_IWES_32m'].attrs['units'] = 'Degrees'
    ds['inflow_angle_IWES_32m'].attrs['height'] = '32m'

    ds['u_IWES_32m'].attrs['description'] = '10min block average of the along-wind speed component. Also called stream-wise wind component. Is the wind component aligned with the main wind direction over a 10-min period by performing a pitch and roll sonic coordinate rotation'
    ds['u_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['u_IWES_32m'].attrs['height'] = '32m'

    ds['v_IWES_32m'].attrs['description'] = '10min block average of the cross-wind speed component. Is the wind component perpendicular to the main wind direction over a 10-min period after pitch and roll rotation'
    ds['v_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['v_IWES_32m'].attrs['height'] = '32m'

    ds['w_IWES_32m'].attrs['description'] = '10min block average of the vertical wind speed component after double rotation. Is the vertical wind component after pitch and roll sonic coordinate rotation in relation to the 10-min mean wind direction. Positive vertical wind is pointing upwards'
    ds['w_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['w_IWES_32m'].attrs['height'] = '32m'

    ds['T_IWES_32m'].attrs['description'] = '10min block average sonic temperature sampled at 20 Hz'
    ds['T_IWES_32m'].attrs['units'] = 'Degrees Kelvin'
    ds['T_IWES_32m'].attrs['height'] = '32m'

    ds['u_max_IWES_32m'].attrs['description'] = 'Maximum along-wind speed component (after double rotation) over 10-min period.'
    ds['u_max_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['u_max_IWES_32m'].attrs['height'] = '32m'

    ds['v_max_IWES_32m'].attrs['description'] = 'Maximum cross-wind component (after double rotation) over 10-min period.'
    ds['v_max_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['v_max_IWES_32m'].attrs['height'] = '32m'

    ds['w_max_IWES_32m'].attrs['description'] = 'Maximum vertical wind component (after double rotation) over 10-min period.'
    ds['w_max_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['w_max_IWES_32m'].attrs['height'] = '32m'

    ds['T_max_IWES_32m'].attrs['description'] = 'Maximum sonic temperature over 10-min period.'
    ds['T_max_IWES_32m'].attrs['units'] = 'Degrees Kelvin'
    ds['T_max_IWES_32m'].attrs['height'] = '32m'

    ds['u_min_IWES_32m'].attrs['description'] = 'Minimum along-wind speed component (after double rotation) over 10-min period.'
    ds['u_min_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['u_min_IWES_32m'].attrs['height'] = '32m'

    ds['v_min_IWES_32m'].attrs['description'] = 'Minimum cross-wind component (after double rotation) over 10-min period.'
    ds['v_min_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['v_min_IWES_32m'].attrs['height'] = '32m'

    ds['w_min_IWES_32m'].attrs['description'] = 'Minimum vertical wind component (after double rotation) over 10-min period.'
    ds['w_min_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['w_min_IWES_32m'].attrs['height'] = '32m'

    ds['T_min_IWES_32m'].attrs['description'] = 'Minimum sonic temperature over 10-min period.'
    ds['T_min_IWES_32m'].attrs['units'] = 'Degrees Kelvin'
    ds['T_min_IWES_32m'].attrs['height'] = '32m'

    ds['cov_uu_IWES_32m'].attrs['description'] = 'Auto-covariance of the along-wind speed component'
    ds['cov_uu_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uu_IWES_32m'].attrs['height'] = '32m'

    ds['cov_uv_IWES_32m'].attrs['description'] = 'Covariance of the along-wind and cross-wind speed components.'
    ds['cov_uv_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uv_IWES_32m'].attrs['height'] = '32m'

    ds['cov_uw_IWES_32m'].attrs['description'] = 'Covariance of the along-wind and vertical wind speed components.'
    ds['cov_uw_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uw_IWES_32m'].attrs['height'] = '32m'

    ds['cov_vv_IWES_32m'].attrs['description'] = 'Auto-covariance of the cross-wind speed component'
    ds['cov_vv_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_vv_IWES_32m'].attrs['height'] = '32m'

    ds['cov_vw_IWES_32m'].attrs['description'] = 'Covariance of the cross-wind and vertical wind speed components.'
    ds['cov_vw_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_vw_IWES_32m'].attrs['height'] = '32m'

    ds['cov_ww_IWES_32m'].attrs['description'] = 'Auto-covariance of the vertical wind speed component'
    ds['cov_ww_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_ww_IWES_32m'].attrs['height'] = '32m'

    ds['cov_uT_IWES_32m'].attrs['description'] = 'Covariance of the along-wind and temperature.'
    ds['cov_uT_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_uT_IWES_32m'].attrs['height'] = '32m'

    ds['cov_vT_IWES_32m'].attrs['description'] = 'Covariance of the cross-wind and temperature.'
    ds['cov_vT_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_vT_IWES_32m'].attrs['height'] = '32m'

    ds['cov_wT_IWES_32m'].attrs['description'] = 'Upward sensible heat flux in the air using sonic temperature'
    ds['cov_wT_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_wT_IWES_32m'].attrs['height'] = '32m'

    ds['cov_TT_IWES_32m'].attrs['description'] = 'Auto-covariance of sonic temperature.'
    ds['cov_TT_IWES_32m'].attrs['units'] = 'Meters^2/second^2'
    ds['cov_TT_IWES_32m'].attrs['height'] = '32m'

    ds['U_horz_std_IWES_32m'].attrs['description'] = 'Standard deviation of the 10min horizontal wind speed sampled at 20Hz.'
    ds['U_horz_std_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['U_horz_std_IWES_32m'].attrs['height'] = '32m'

    ds['U_vec_std_IWES_32m'].attrs['description'] = 'Standard deviation of the 10min magnitude of the 3D wind vector sampled at 20Hz.'
    ds['U_vec_std_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['U_vec_std_IWES_32m'].attrs['height'] = '32m'

    ds['u_star_IWES_32m'].attrs['description'] = '10min block average Friction velocity sampled over 20Hz with along- and cross-wind components'
    ds['u_star_IWES_32m'].attrs['units'] = 'Meters/second'
    ds['u_star_IWES_32m'].attrs['height'] = '32m'

    ds['L-1_IWES_32m'].attrs['description'] = 'Inverse of the Obukhov Length'
    ds['L-1_IWES_32m'].attrs['units'] = 'Meters^-1'
    ds['L-1_IWES_32m'].attrs['height'] = '32m'

    ds['zL_IWES_32m'].attrs['description'] = 'Dimensionless stability parameter, where the Obukhov Length is scaled by the measurement sonic height'
    ds['zL_IWES_32m'].attrs['units'] = 'None'
    ds['zL_IWES_32m'].attrs['height'] = '32m'

    # Sort dataset alphabetically by data variable names
    ds = ds[sorted(ds.data_vars)]

    return ds


def save_file(df, ds, outpath):
    # Make dict for compression encoding
    # Example, raw data files = 745 MB. No compression = 980 MB. zlib compression = 628 MB
    encode_dict = {
        'RECORD': {'zlib': True},
        'NumSamples': {'zlib': True},
        'DL_IntTemp_Avg': {'zlib': True},
        'SupplyVolt_Avg': {'zlib': True},
        'SkppdRecord1': {'zlib': True},
        'SkppdRecord2': {'zlib': True},
        'NTP_PingServerResp': {'zlib': True},
        'NTP_TimeOffset': {'zlib': True},
        'SkppdScan': {'zlib': True},
        'SkppdSlwScan1': {'zlib': True},
        'SkppdSlwScan2': {'zlib': True},
        'SkppdSlwScan3': {'zlib': True},
        'SkppdSlwScan4': {'zlib': True},
        'NTP_DtTmSync': {'zlib': True},
        'Vane_32m_NW_Dir_1Hz_Raw_Avg': {'zlib': True},
        'Temp_96m_TC_1Hz_Raw_Avg': {'zlib': True},
        'Temp_CR_TC_1Hz_Raw_Avg': {'zlib': True},
        'RHS_96m_RH_1Hz_Raw_Avg': {'zlib': True},
        'RHS_CR_RH_1Hz_Raw_Avg': {'zlib': True},
        'Pres_96m_PR_1Hz_Raw_Avg': {'zlib': True},
        'Pres_C_PR_1Hz_Raw_Avg': {'zlib': True},
        'TempD_CR_96m_dTC_1Hz_Raw_Avg': {'zlib': True},
        'SST01_Sky_MD_1Hz_Raw_Avg': {'zlib': True},
        'SST01_Sea_MD_1Hz_Raw_Avg': {'zlib': True},
        'Aneo_96m_N_Spd_1Hz_Calc_Avg': {'zlib': True},
        'Aneo_96m_S_Spd_1Hz_Calc_Avg': {'zlib': True},
        'Aneo_92m_NW_Spd_1Hz_Calc_Avg': {'zlib': True},
        'Aneo_70m_NW_Spd_1Hz_Calc_Avg': {'zlib': True},
        'Aneo_48m_NW_Spd_1Hz_Calc_Avg': {'zlib': True},
        'Aneo_32m_NW_Spd_1Hz_Calc_Avg': {'zlib': True},
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
        'SST01_Sea_1Hz_RadEm_Calc_Avg': {'zlib': True},
        'SST_SkyCorrection_1Hz_Calc_Avg': {'zlib': True},
        'Vane_32m_NW_Dir_1Hz_Raw_Avg_Min': {'zlib': True},
        'Temp_96m_TC_1Hz_Raw_Avg_Min': {'zlib': True},
        'Temp_CR_TC_1Hz_Raw_Avg_Min': {'zlib': True},
        'RHS_96m_RH_1Hz_Raw_Avg_Min': {'zlib': True},
        'RHS_CR_RH_1Hz_Raw_Avg_Min': {'zlib': True},
        'Pres_96m_PR_1Hz_Raw_Avg_Min': {'zlib': True},
        'Pres_C_PR_1Hz_Raw_Avg_Min': {'zlib': True},
        'TempD_CR_96m_dTC_1Hz_Raw_Avg_Min': {'zlib': True},
        'SST01_Sky_MD_1Hz_Raw_Avg_Min': {'zlib': True},
        'SST01_Sea_MD_1Hz_Raw_Avg_Min': {'zlib': True},
        'Aneo_96m_N_Spd_1Hz_Calc_Avg_Min': {'zlib': True},
        'Aneo_96m_S_Spd_1Hz_Calc_Avg_Min': {'zlib': True},
        'Aneo_92m_NW_Spd_1Hz_Calc_Avg_Min': {'zlib': True},
        'Aneo_70m_NW_Spd_1Hz_Calc_Avg_Min': {'zlib': True},
        'Aneo_48m_NW_Spd_1Hz_Calc_Avg_Min': {'zlib': True},
        'Aneo_32m_NW_Spd_1Hz_Calc_Avg_Min': {'zlib': True},
        'USA3D_92m_S_x_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_92m_S_y_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_92m_S_z_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_92m_S_FloTC_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_32m_S_x_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_32m_S_y_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_32m_S_z_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_32m_S_FloTC_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Raw_Avg_Min': {'zlib': True},
        'USA3D_92m_S_Spd_8Hz_Calc_Avg_Min': {'zlib': True},
        'USA3D_32m_S_Spd_8Hz_Calc_Avg_Min': {'zlib': True},
        'USA3D_92m_S_Dir_8Hz_Calc_Avg_Min': {'zlib': True},
        'USA3D_32m_S_Dir_8Hz_Calc_Avg_Min': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Calc_Avg_Min': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Calc_Avg_Min': {'zlib': True},
        'SST01_Sea_1Hz_RadEm_Calc_Avg_Min': {'zlib': True},
        'SST_SkyCorrection_1Hz_Calc_Avg_Min': {'zlib': True},
        'Vane_32m_NW_Dir_1Hz_Raw_Avg_Max': {'zlib': True},
        'Temp_96m_TC_1Hz_Raw_Avg_Max': {'zlib': True},
        'Temp_CR_TC_1Hz_Raw_Avg_Max': {'zlib': True},
        'RHS_96m_RH_1Hz_Raw_Avg_Max': {'zlib': True},
        'RHS_CR_RH_1Hz_Raw_Avg_Max': {'zlib': True},
        'Pres_96m_PR_1Hz_Raw_Avg_Max': {'zlib': True},
        'Pres_C_PR_1Hz_Raw_Avg_Max': {'zlib': True},
        'TempD_CR_96m_dTC_1Hz_Raw_Avg_Max': {'zlib': True},
        'SST01_Sky_MD_1Hz_Raw_Avg_Max': {'zlib': True},
        'SST01_Sea_MD_1Hz_Raw_Avg_Max': {'zlib': True},
        'Aneo_96m_N_Spd_1Hz_Calc_Avg_Max': {'zlib': True},
        'Aneo_96m_S_Spd_1Hz_Calc_Avg_Max': {'zlib': True},
        'Aneo_92m_NW_Spd_1Hz_Calc_Avg_Max': {'zlib': True},
        'Aneo_70m_NW_Spd_1Hz_Calc_Avg_Max': {'zlib': True},
        'Aneo_48m_NW_Spd_1Hz_Calc_Avg_Max': {'zlib': True},
        'Aneo_32m_NW_Spd_1Hz_Calc_Avg_Max': {'zlib': True},
        'USA3D_92m_S_x_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_92m_S_y_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_92m_S_z_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_92m_S_FloTC_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_32m_S_x_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_32m_S_y_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_32m_S_z_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_32m_S_FloTC_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Raw_Avg_Max': {'zlib': True},
        'USA3D_92m_S_Spd_8Hz_Calc_Avg_Max': {'zlib': True},
        'USA3D_32m_S_Spd_8Hz_Calc_Avg_Max': {'zlib': True},
        'USA3D_92m_S_Dir_8Hz_Calc_Avg_Max': {'zlib': True},
        'USA3D_32m_S_Dir_8Hz_Calc_Avg_Max': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Calc_Avg_Max': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Calc_Avg_Max': {'zlib': True},
        'SST01_Sea_1Hz_RadEm_Calc_Avg_Max': {'zlib': True},
        'SST_SkyCorrection_1Hz_Calc_Avg_Max': {'zlib': True},
        'Vane_32m_NW_Dir_1Hz_Raw_Avg_Std': {'zlib': True},
        'Temp_96m_TC_1Hz_Raw_Avg_Std': {'zlib': True},
        'Temp_CR_TC_1Hz_Raw_Avg_Std': {'zlib': True},
        'RHS_96m_RH_1Hz_Raw_Avg_Std': {'zlib': True},
        'RHS_CR_RH_1Hz_Raw_Avg_Std': {'zlib': True},
        'Pres_96m_PR_1Hz_Raw_Avg_Std': {'zlib': True},
        'Pres_C_PR_1Hz_Raw_Avg_Std': {'zlib': True},
        'TempD_CR_96m_dTC_1Hz_Raw_Avg_Std': {'zlib': True},
        'SST01_Sky_MD_1Hz_Raw_Avg_Std': {'zlib': True},
        'SST01_Sea_MD_1Hz_Raw_Avg_Std': {'zlib': True},
        'Aneo_96m_N_Spd_1Hz_Calc_Avg_Std': {'zlib': True},
        'Aneo_96m_S_Spd_1Hz_Calc_Avg_Std': {'zlib': True},
        'Aneo_92m_NW_Spd_1Hz_Calc_Avg_Std': {'zlib': True},
        'Aneo_70m_NW_Spd_1Hz_Calc_Avg_Std': {'zlib': True},
        'Aneo_48m_NW_Spd_1Hz_Calc_Avg_Std': {'zlib': True},
        'Aneo_32m_NW_Spd_1Hz_Calc_Avg_Std': {'zlib': True},
        'USA3D_92m_S_x_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_92m_S_y_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_92m_S_z_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_92m_S_FloTC_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_32m_S_x_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_32m_S_y_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_32m_S_z_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_32m_S_FloTC_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Raw_Avg_Std': {'zlib': True},
        'USA3D_92m_S_Spd_8Hz_Calc_Avg_Std': {'zlib': True},
        'USA3D_32m_S_Spd_8Hz_Calc_Avg_Std': {'zlib': True},
        'USA3D_92m_S_Dir_8Hz_Calc_Avg_Std': {'zlib': True},
        'USA3D_32m_S_Dir_8Hz_Calc_Avg_Std': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Calc_Avg_Std': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Calc_Avg_Std': {'zlib': True},
        'SST01_Sea_1Hz_RadEm_Calc_Avg_Std': {'zlib': True},
        'SST_SkyCorrection_1Hz_Calc_Avg_Std': {'zlib': True},
        'U_horz_IWES_92m': {'zlib': True},
        'U_vec_IWES_92m': {'zlib': True},
        'wind_direction_IWES_92m': {'zlib': True},
        'inflow_angle_IWES_92m': {'zlib': True},
        'u_IWES_92m': {'zlib': True},
        'v_IWES_92m': {'zlib': True},
        'w_IWES_92m': {'zlib': True},
        'T_IWES_92m': {'zlib': True},
        'u_max_IWES_92m': {'zlib': True},
        'v_max_IWES_92m': {'zlib': True},
        'w_max_IWES_92m': {'zlib': True},
        'T_max_IWES_92m': {'zlib': True},
        'u_min_IWES_92m': {'zlib': True},
        'v_min_IWES_92m': {'zlib': True},
        'w_min_IWES_92m': {'zlib': True},
        'T_min_IWES_92m': {'zlib': True},
        'cov_uu_IWES_92m': {'zlib': True},
        'cov_uv_IWES_92m': {'zlib': True},
        'cov_uw_IWES_92m': {'zlib': True},
        'cov_vv_IWES_92m': {'zlib': True},
        'cov_vw_IWES_92m': {'zlib': True},
        'cov_ww_IWES_92m': {'zlib': True},
        'cov_uT_IWES_92m': {'zlib': True},
        'cov_vT_IWES_92m': {'zlib': True},
        'cov_wT_IWES_92m': {'zlib': True},
        'cov_TT_IWES_92m': {'zlib': True},
        'U_horz_std_IWES_92m': {'zlib': True},
        'U_vec_std_IWES_92m': {'zlib': True},
        'u_star_IWES_92m': {'zlib': True},
        'L-1_IWES_92m': {'zlib': True},
        'zL_IWES_92m': {'zlib': True},
        'U_horz_IWES_32m': {'zlib': True},
        'U_vec_IWES_32m': {'zlib': True},
        'wind_direction_IWES_32m': {'zlib': True},
        'inflow_angle_IWES_32m': {'zlib': True},
        'u_IWES_32m': {'zlib': True},
        'v_IWES_32m': {'zlib': True},
        'w_IWES_32m': {'zlib': True},
        'T_IWES_32m': {'zlib': True},
        'u_max_IWES_32m': {'zlib': True},
        'v_max_IWES_32m': {'zlib': True},
        'w_max_IWES_32m': {'zlib': True},
        'T_max_IWES_32m': {'zlib': True},
        'u_min_IWES_32m': {'zlib': True},
        'v_min_IWES_32m': {'zlib': True},
        'w_min_IWES_32m': {'zlib': True},
        'T_min_IWES_32m': {'zlib': True},
        'cov_uu_IWES_32m': {'zlib': True},
        'cov_uv_IWES_32m': {'zlib': True},
        'cov_uw_IWES_32m': {'zlib': True},
        'cov_vv_IWES_32m': {'zlib': True},
        'cov_vw_IWES_32m': {'zlib': True},
        'cov_ww_IWES_32m': {'zlib': True},
        'cov_uT_IWES_32m': {'zlib': True},
        'cov_vT_IWES_32m': {'zlib': True},
        'cov_wT_IWES_32m': {'zlib': True},
        'cov_TT_IWES_32m': {'zlib': True},
        'U_horz_std_IWES_32m': {'zlib': True},
        'U_vec_std_IWES_32m': {'zlib': True},
        'u_star_IWES_32m': {'zlib': True},
        'L-1_IWES_32m': {'zlib': True},
        'zL_IWES_32m': {'zlib': True}
    }

    fname = 'NSO-met-mast-data-10min_' + str(df.first_valid_index().strftime('%Y-%m-%d-%H-%M-%S')) + '_' + str(
        df.last_valid_index().strftime('%Y-%m-%d-%H-%M-%S'))
    print(fname)

    ds.to_netcdf(str(outpath) + '\\' + fname + '.nc', format='NETCDF4', engine='netcdf4')


def main():
    infile, outpath, path_1Hz, path_iwes_92m, path_iwes_32m = init()
    df = loadfiles(infile)
    df = format_df(df, path_1Hz, path_iwes_92m, path_iwes_32m)
    ds = format_xds(df)
    ds = add_metadata(ds)
    save_file(df, ds, outpath)
    print('Done!')


if __name__ == '__main__':
    main()