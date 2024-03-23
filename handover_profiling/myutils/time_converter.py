#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import datetime as dt

__all__ = [
    "datetime_to_str",
    "str_to_datetime",
    "epoch_to_datetime",
    "datetime_to_epoch",
]

def datetime_to_str(timestamp_datetime):
    return dt.datetime.strftime(timestamp_datetime, "%Y-%m-%d %H:%M:%S.%f")

def str_to_datetime(timestamp_str, format='pd'):
    if format == 'pd':
        return pd.to_datetime(timestamp_str)
    elif format == 'dt':
        return dt.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

def epoch_to_datetime(timestamp_epoch, format='pd', utc=8):
    if format == 'pd':
        return pd.to_datetime(timestamp_epoch, unit='s') + pd.Timedelta(hours=utc)
    elif format == 'dt':
        return dt.datetime.utcfromtimestamp(timestamp_epoch) + dt.timedelta(hours=utc)

def datetime_to_epoch(timestamp_datetime, utc=8):
    # Set the timezone
    timezone = dt.timezone(dt.timedelta(hours=utc))
    timestamp_datetime = timestamp_datetime.replace(tzinfo=timezone)
    # Convert the datetime object to Unix timestamp
    return timestamp_datetime.timestamp()
