"""Module to download analysis data"""
import logging
import os

from gwpy.timeseries import TimeSeries

TRIGGER_TIME = 1135136350.6
GPS_START_TIME = TRIGGER_TIME - 2
DATA_DURATION = 4
EXTRA_DATA = 10
DATA_START_TIME = int(GPS_START_TIME) - EXTRA_DATA
DATA_END_TIME = int(GPS_START_TIME + DATA_DURATION) + EXTRA_DATA

CHANNELS = dict(
    L1='L1:DCS-CALIB_STRAIN_C02',
    H1='H1:DCS-CALIB_STRAIN_C02'
)

DATA_FOLDER = "."
DATA_FILE_NAME = os.path.join(DATA_FOLDER, "{ifo}_data.gwf")


def download_data():
    for ifo, channel in CHANNELS.items():
        logging.info(f"Downloading {channel} data...")
        data = TimeSeries.get(channel, DATA_START_TIME, DATA_END_TIME)
        TimeSeries.write(data, target=DATA_FILE_NAME.format(ifo=ifo), format='gwf')
    logging.info("Data downloading complete")


def main():
    download_data()


if __name__ == '__main__':
    main()

