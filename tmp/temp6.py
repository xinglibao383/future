import h5py

# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/tmp/temp6.py
if __name__ == "__main__":
    file_path = '/mnt/mydata/yh/liming/data/xrfv2/wifi/12_2_15.h5'
    with h5py.File(file_path, 'r') as f:
        print(f"{f['label'][()]}")
    print("================================================================================")
    file_path = '/mnt/mydata/yh/liming/data/xrfv2/imu/12_2_15.h5'
    with h5py.File(file_path, 'r') as f:
        print(f"{f['label'][()]}")
    