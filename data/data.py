import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.preprocessing import (
    QuantileTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import torch
import argparse
from collections import namedtuple, OrderedDict


def get_nuclear_data():
    def lc_read_csv(url):
        req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
        )
        return pd.read_csv(urllib.request.urlopen(req))

    if not os.path.exists("data"):
        os.mkdir("data")
    else:
        df2 = pd.read_csv("data/ame2020.csv").set_index(["z", "n"])
        df2 = df2[~df2.index.duplicated(keep="first")]
        df = pd.read_csv("data/ground_states.csv").set_index(["z", "n"])
        df["binding_unc"] = df2.binding_unc
        df["binding_sys"] = df2.binding_sys
        df.reset_index(inplace=True)
    
    # Clean up data points
    df['binding'] = [float(b) if ' ' not in b else 0.0 for b in df['binding']]
    
    # Change to float
    df['binding'] = df['binding'].astype(float)
    df['binding_unc'] = df['binding_unc'].astype(float)
    
    # Get rid of uncertain data points
    df = df[df['binding_unc'] <= (0.01 * df['binding'])]
    
    # Get rid of 'per-nucleon' and convert from keV to MeV
    df['binding'] = (df['binding'] * (df['n'] + df['z'])) / 1000.0
    df['binding_unc'] = (df['binding_unc'] * (df['n'] + df['z'])) / 1000.0
    
    df.reset_index(inplace=True)
    return df[['z', 'n', 'binding']]