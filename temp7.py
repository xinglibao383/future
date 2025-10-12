import os
import json
import h5py
import numpy as np

with h5py.File('/data/luohonglin/imu/1_3_5.h5', 'r') as f:
    data = f['data'][()]
print(data.shape)

