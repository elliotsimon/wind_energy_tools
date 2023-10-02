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


def init(infile=sys.argv[1], outpath=sys.argv[2]):
    #print(Path(infile), Path(outpath))
    return Path(infile), Path(outpath)


def load_files(infile):
    '''
    Input: Single .DAT file from met-mast data logger.
    Output: formatted pandas DataFrame containing measurement data.
    '''
    # Keep the channel header but skip the initialization info and units
    print(infile)
    infile = [str(infile)]
    rowlist = [0, 2, 3]
    df = pd.concat(
        map(lambda f: pd.read_csv(f, skiprows=rowlist, header=0, na_values='NAN', encoding='unicode-escape', on_bad_lines='skip'), infile))
    print('File read!')
    # Set the index as our datetime stamp
    df.set_index(pd.to_datetime(df['TIMESTAMP']), inplace=True)
    del df['TIMESTAMP']
    # Convert objects to correct dtypes
    df = df.convert_dtypes()

    # Fix status channels
    df['USA3D_92m_S_Status_8Hz_Raw'] = df['USA3D_92m_S_Status_8Hz_Raw'].fillna(np.nan)
    df['USA3D_32m_S_Status_8Hz_Raw'] = df['USA3D_32m_S_Status_8Hz_Raw'].fillna(np.nan)

    # Map dict of status flag. Replace M values with True. Join back to original df and drop string columns
    status_dict = {'M': True, '': False, 'NA': False, 'NAN': False, np.nan: False}
    df = df.join(df.select_dtypes(exclude=['object', 'int', 'float']).replace(status_dict).fillna(False),
                 lsuffix='_DROP')
    df = df.loc[:, ~df.columns.str.contains('DROP')]
    # Force status channels to bool then int
    df['USA3D_92m_S_Status_8Hz_Raw'] = df['USA3D_92m_S_Status_8Hz_Raw'].astype(bool).astype(int)
    df['USA3D_32m_S_Status_8Hz_Raw'] = df['USA3D_32m_S_Status_8Hz_Raw'].astype(bool).astype(int)

    df = df.sort_index()
    return df


def outlier_removal(s, threshold=6, window='5S'):
    # Moving median outlier removal
    median = s.rolling(window).median()
    std = s.rolling(window).std()
    s = s[(s <= median + threshold * std) & (s >= median - threshold * std)]
    s = s.interpolate(method='ffill', limit=1)
    s = s.interpolate(method='bfill', limit=1)
    # Average values with duplicated indices
    s = s.groupby(s.index).mean()
    return s


def process_df(df, sonic_32m_dir_offset, sonic_92m_dir_offset):
    # Rename columns
    df = df.rename(columns={"USA3D_92m_S_u_8Hz_Raw": "USA3D_92m_S_x_8Hz_Raw",
                            "USA3D_92m_S_v_8Hz_Raw": "USA3D_92m_S_y_8Hz_Raw",
                            "USA3D_92m_S_w_8Hz_Raw": "USA3D_92m_S_z_8Hz_Raw",
                            "USA3D_32m_S_u_8Hz_Raw": "USA3D_32m_S_x_8Hz_Raw",
                            "USA3D_32m_S_v_8Hz_Raw": "USA3D_32m_S_y_8Hz_Raw",
                            "USA3D_32m_S_w_8Hz_Raw": "USA3D_32m_S_z_8Hz_Raw"})

    # Convert from cm/s to m/s
    df['USA3D_92m_S_x_8Hz_Raw'] /= 100
    df['USA3D_92m_S_y_8Hz_Raw'] /= 100
    df['USA3D_92m_S_z_8Hz_Raw'] /= 100
    df['USA3D_32m_S_x_8Hz_Raw'] /= 100
    df['USA3D_32m_S_y_8Hz_Raw'] /= 100
    df['USA3D_32m_S_z_8Hz_Raw'] /= 100

    # Apply outlier removal filter
    df['USA3D_92m_S_x_8Hz_Raw'] = outlier_removal(df['USA3D_92m_S_x_8Hz_Raw'].dropna())
    df['USA3D_92m_S_y_8Hz_Raw'] = outlier_removal(df['USA3D_92m_S_y_8Hz_Raw'].dropna())
    df['USA3D_92m_S_z_8Hz_Raw'] = outlier_removal(df['USA3D_92m_S_z_8Hz_Raw'].dropna())
    df['USA3D_32m_S_x_8Hz_Raw'] = outlier_removal(df['USA3D_32m_S_x_8Hz_Raw'].dropna())
    df['USA3D_32m_S_y_8Hz_Raw'] = outlier_removal(df['USA3D_32m_S_y_8Hz_Raw'].dropna())
    df['USA3D_32m_S_z_8Hz_Raw'] = outlier_removal(df['USA3D_32m_S_z_8Hz_Raw'].dropna())

    # Delete old (wrong) sonic channels
    del df['USA3D_92m_S_Spd_8Hz_Raw']
    del df['USA3D_32m_S_Spd_8Hz_Raw']
    del df['USA3D_92m_S_Dir_8Hz_Raw']
    del df['USA3D_32m_S_Dir_8Hz_Raw']

    # Re-do reconstruction of sonic speed and direction
    # 3D reconstuction (x,y,z)
    df['USA3D_92m_S_Spd_8Hz_Calc'] = np.sqrt(
        df['USA3D_92m_S_x_8Hz_Raw'] ** 2 + df['USA3D_92m_S_y_8Hz_Raw'] ** 2 + df['USA3D_92m_S_z_8Hz_Raw'] ** 2)
    df['USA3D_32m_S_Spd_8Hz_Calc'] = np.sqrt(
        df['USA3D_32m_S_x_8Hz_Raw'] ** 2 + df['USA3D_32m_S_y_8Hz_Raw'] ** 2 + df['USA3D_32m_S_z_8Hz_Raw'] ** 2)
    # Is there an offset to apply to wind direction???
    df['USA3D_92m_S_Dir_8Hz_Calc'] = (np.rad2deg(
        np.arctan2(df['USA3D_92m_S_y_8Hz_Raw'], df['USA3D_92m_S_x_8Hz_Raw'])) + 360) % 360
    df['USA3D_32m_S_Dir_8Hz_Calc'] = (np.rad2deg(
        np.arctan2(df['USA3D_32m_S_y_8Hz_Raw'], df['USA3D_32m_S_x_8Hz_Raw'])) + 360) % 360

    # Apply offsets to wind direction
    df['USA3D_32m_S_Dir_8Hz_Calc'] = (df['USA3D_32m_S_Dir_8Hz_Calc'] + sonic_32m_dir_offset + 360) % 360
    df['USA3D_92m_S_Dir_8Hz_Calc'] = (df['USA3D_92m_S_Dir_8Hz_Calc'] + sonic_92m_dir_offset + 360) % 360

    # Create new calculated flow inclination columns
    df['USA3D_92m_S_Inc_8Hz_Calc'] = np.rad2deg(np.arctan2(df['USA3D_92m_S_z_8Hz_Raw'], np.sqrt(
        df['USA3D_92m_S_x_8Hz_Raw'] ** 2 + df['USA3D_92m_S_y_8Hz_Raw'] ** 2)))
    df['USA3D_32m_S_Inc_8Hz_Calc'] = np.rad2deg(np.arctan2(df['USA3D_32m_S_z_8Hz_Raw'], np.sqrt(
        df['USA3D_32m_S_x_8Hz_Raw'] ** 2 + df['USA3D_32m_S_y_8Hz_Raw'] ** 2)))

    # Convert pandas._libs.missing.NAType to np.nan
    df.fillna(np.nan, inplace=True)

    return df


def process_xds(df):
    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(df)
    # Ensure missing data is correct format for netcdf file (must be np.nan (float object) and not pandas.NaT)
    ds = ds.fillna(np.nan)

    return ds


def ds_attrributes(ds):
    # Add attributes (units, channel description, height etc.)
    ds.attrs['Author'] = 'Elliot Simon'
    ds.attrs['Contact'] = 'ellsim@dtu.dk'
    ds.attrs['Project'] = 'OWA GLoBE'
    ds.attrs['Position_UTM'] = '411734.4371 m E, 6028967.271 m N, UTM WGS84 Zone32'
    ds.attrs['Time_Zone'] = 'UTC, no DST shift'
    ds.attrs['NTP_Sync_Enabled'] = 'True'
    ds.attrs['Date_Processed'] = str(datetime.datetime.now())

    ds['RECORD'].attrs['description'] = 'Sample index'
    ds['RECORD'].attrs['units'] = 'Integer index'
    ds['RECORD'].attrs['height'] = 'N/A'

    ds['USA3D_92m_S_x_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) x-axis speed at 92m height. Outliers removed using moving-median filter'
    ds['USA3D_92m_S_x_8Hz_Raw'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_x_8Hz_Raw'].attrs['height'] = '92m'

    ds['USA3D_92m_S_y_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) y-axis speed at 92m height. Outliers removed using moving-median filter'
    ds['USA3D_92m_S_y_8Hz_Raw'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_y_8Hz_Raw'].attrs['height'] = '92m'

    ds['USA3D_92m_S_z_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) z-axis speed at 92m height. Outliers removed using moving-median filter'
    ds['USA3D_92m_S_z_8Hz_Raw'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_z_8Hz_Raw'].attrs['height'] = '92m'

    ds['USA3D_92m_S_FloTC_8Hz_Raw'].attrs['description'] = 'Sonic anemometer (South boom) temperature at 92m height'
    ds['USA3D_92m_S_FloTC_8Hz_Raw'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_92m_S_FloTC_8Hz_Raw'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Spd_8Hz_Calc'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind speed at 92m height calculated using 3D reconstruction: sqrt(x^2+y^2+z^2)'
    ds['USA3D_92m_S_Spd_8Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['USA3D_92m_S_Spd_8Hz_Calc'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Dir_8Hz_Calc'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind direction at 92m height calculated using 2D reconstruction: (rad2deg(arctan2(y,x))+360)%360 with offset applied'
    ds['USA3D_92m_S_Dir_8Hz_Calc'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Dir_8Hz_Calc'].attrs['height'] = '92m'

    ds['USA3D_92m_S_Inc_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 92m height, raw output'
    ds['USA3D_92m_S_Inc_8Hz_Raw'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Raw'].attrs['height'] = '92m'

    ds['USA3D_32m_S_x_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) x-axis speed at 32m height. Outliers removed using moving-median filter'
    ds['USA3D_32m_S_x_8Hz_Raw'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_x_8Hz_Raw'].attrs['height'] = '32m'

    ds['USA3D_32m_S_y_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) y-axis speed at 32m height. Outliers removed using moving-median filter'
    ds['USA3D_32m_S_y_8Hz_Raw'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_y_8Hz_Raw'].attrs['height'] = '32m'

    ds['USA3D_32m_S_z_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) z-axis speed at 32m height. Outliers removed using moving-median filter'
    ds['USA3D_32m_S_z_8Hz_Raw'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_z_8Hz_Raw'].attrs['height'] = '32m'

    ds['USA3D_32m_S_FloTC_8Hz_Raw'].attrs['description'] = 'Sonic anemometer (South boom) temperature at 32m height'
    ds['USA3D_32m_S_FloTC_8Hz_Raw'].attrs['units'] = 'Degrees Celsius'
    ds['USA3D_32m_S_FloTC_8Hz_Raw'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Spd_8Hz_Calc'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind speed at 32m height calculated using 3D reconstruction: sqrt(x^2+y^2+z^2)'
    ds['USA3D_32m_S_Spd_8Hz_Calc'].attrs['units'] = 'Meters/second'
    ds['USA3D_32m_S_Spd_8Hz_Calc'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Dir_8Hz_Calc'].attrs[
        'description'] = 'Sonic anemometer (South boom) wind direction at 32m height calculated using 2D reconstruction: (rad2deg(arctan2(y,x))+360)%360 with offset applied'
    ds['USA3D_32m_S_Dir_8Hz_Calc'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Dir_8Hz_Calc'].attrs['height'] = '32m'

    ds['USA3D_32m_S_Inc_8Hz_Raw'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 32m height, raw output'
    ds['USA3D_32m_S_Inc_8Hz_Raw'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Raw'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Status_8Hz_Raw'].attrs['description'] = 'Sonic anemometer (South boom) status flag at 92m height'
    ds['USA3D_92m_S_Status_8Hz_Raw'].attrs['units'] = 'Boolean Flag'
    ds['USA3D_92m_S_Status_8Hz_Raw'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Status_8Hz_Raw'].attrs['description'] = 'Sonic anemometer (South boom) status flag at 32m height'
    ds['USA3D_32m_S_Status_8Hz_Raw'].attrs['units'] = 'Boolean Flag'
    ds['USA3D_32m_S_Status_8Hz_Raw'].attrs['height'] = '32m'

    ds['USA3D_92m_S_Inc_8Hz_Calc'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 92m height calculated using: arctan2(z,sqrt(x^2+y^2))'
    ds['USA3D_92m_S_Inc_8Hz_Calc'].attrs['units'] = 'Degrees'
    ds['USA3D_92m_S_Inc_8Hz_Calc'].attrs['height'] = '92m'

    ds['USA3D_32m_S_Inc_8Hz_Calc'].attrs[
        'description'] = 'Sonic anemometer (South boom) flow inclination at 32m height calculated using: arctan2(z,sqrt(x^2+y^2))'
    ds['USA3D_32m_S_Inc_8Hz_Calc'].attrs['units'] = 'Degrees'
    ds['USA3D_32m_S_Inc_8Hz_Calc'].attrs['height'] = '32m'

    return ds


def save_ds(ds, df, outpath):
    # Make dict for compression encoding
    # Example, raw data files = 745 MB. No compression = 980 MB. zlib compression = 628 MB
    encode_dict = {
        'RECORD': {'zlib': True},
        'USA3D_92m_S_x_8Hz_Raw': {'zlib': True},
        'USA3D_92m_S_y_8Hz_Raw': {'zlib': True},
        'USA3D_92m_S_z_8Hz_Raw': {'zlib': True},
        'USA3D_92m_S_FloTC_8Hz_Raw': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Raw': {'zlib': True},
        'USA3D_32m_S_x_8Hz_Raw': {'zlib': True},
        'USA3D_32m_S_y_8Hz_Raw': {'zlib': True},
        'USA3D_32m_S_z_8Hz_Raw': {'zlib': True},
        'USA3D_32m_S_FloTC_8Hz_Raw': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Raw': {'zlib': True},
        'USA3D_92m_S_Status_8Hz_Raw': {'zlib': True},
        'USA3D_32m_S_Status_8Hz_Raw': {'zlib': True},
        'USA3D_92m_S_Spd_8Hz_Calc': {'zlib': True},
        'USA3D_32m_S_Spd_8Hz_Calc': {'zlib': True},
        'USA3D_92m_S_Dir_8Hz_Calc': {'zlib': True},
        'USA3D_32m_S_Dir_8Hz_Calc': {'zlib': True},
        'USA3D_92m_S_Inc_8Hz_Calc': {'zlib': True},
        'USA3D_32m_S_Inc_8Hz_Calc': {'zlib': True}}

    fname = 'NSO-met-mast-data-8Hz_' + str(df.first_valid_index().strftime('%Y-%m-%d-%H-%M-%S')) + '_' + str(
        df.last_valid_index().strftime('%Y-%m-%d-%H-%M-%S'))
    print('..Writing to disk..')
    ds.to_netcdf(str(outpath) + '\\' + fname + '.nc', format='NETCDF4', engine='netcdf4', encoding=encode_dict)
    print(fname+'.nc')

    return fname


def plot(ds, outpath, fname):
    f, axarr = plt.subplots(10, 1, sharex=True, figsize=(15, 30))
    plt.sca(axarr[0])
    ds['USA3D_92m_S_Spd_8Hz_Calc'].plot(c='k')
    plt.title('Selected NSO Met-Mast Channels')
    plt.sca(axarr[1])
    ds['USA3D_32m_S_Spd_8Hz_Calc'].plot(c='c')
    plt.sca(axarr[2])
    ds['USA3D_92m_S_Dir_8Hz_Calc'].plot(c='m')
    plt.sca(axarr[3])
    ds['USA3D_32m_S_Dir_8Hz_Calc'].plot(c='y')
    plt.sca(axarr[4])
    ds['USA3D_92m_S_FloTC_8Hz_Raw'].plot(c='b')
    plt.sca(axarr[5])
    ds['USA3D_32m_S_FloTC_8Hz_Raw'].plot(c='r')
    plt.sca(axarr[6])
    ds['USA3D_92m_S_Inc_8Hz_Calc'].plot(c='g')
    plt.sca(axarr[7])
    ds['USA3D_32m_S_Inc_8Hz_Calc'].plot(c='brown')
    plt.sca(axarr[8])
    ds['USA3D_92m_S_Status_8Hz_Raw'].plot(c='orange')
    plt.sca(axarr[9])
    ds['USA3D_32m_S_Status_8Hz_Raw'].plot(c='lime')
    plt.savefig(str(outpath) + '\\plots\\' + fname + '.png', dpi=150)


def main():
    sonic_32m_dir_offset = 208.52
    sonic_92m_dir_offset = 211.66

    # Initialize parameters
    infile, outpath = init()
    df = load_files(infile)
    df = process_df(df, sonic_32m_dir_offset, sonic_92m_dir_offset)
    ds = process_xds(df)
    ds = ds_attrributes(ds)
    fname = save_ds(ds, df, outpath)
    #plot(ds, outpath, fname)
    print('Done!')


if __name__ == '__main__':
    main()

