import h5py

# /home/yh/.conda/envs/myfuture/bin/python /mnt/mydata/yh/liming/workspace/future/tmp/temp5.py
if __name__ == "__main__":
    print("================================================================================")
    file_path = '/mnt/mydata/yh/liming/data/xrfv2/wifi/10_2_7.h5'
    with h5py.File(file_path, 'r') as f:
        print("Keys in the file:", list(f.keys()))

        print(f"Shape of dataset {'amp'}: {f['amp'].shape}")
        print(f"Shape of dataset {'pha'}: {f['pha'].shape}")
        print(f"Shape of dataset {'label'}: {f['label'].shape}")
    print("================================================================================")
    file_path = '/mnt/mydata/yh/liming/data/xrfv2/wifi/13_2_13.h5'
    with h5py.File(file_path, 'r') as f:
        print("Keys in the file:", list(f.keys()))

        print(f"Shape of dataset {'amp'}: {f['amp'].shape}")
        print(f"Shape of dataset {'pha'}: {f['pha'].shape}")
        print(f"Shape of dataset {'label'}: {f['label'].shape}")
    print("================================================================================")
    file_path = '/mnt/mydata/yh/liming/data/xrfv2/imu/10_2_7.h5'
    with h5py.File(file_path, 'r') as f:
        print("Keys in the file:", list(f.keys()))

        print(f"Shape of dataset {'data'}: {f['data'].shape}")
        print(f"Dataset {'duration'}: {f['duration'][()]}")
        print(f"Shape of dataset {'label'}: {f['label'].shape}")
    print("================================================================================")
    file_path = '/mnt/mydata/yh/liming/data/xrfv2/imu/13_2_13.h5'
    with h5py.File(file_path, 'r') as f:
        print("Keys in the file:", list(f.keys()))

        print(f"Shape of dataset {'data'}: {f['data'].shape}")
        print(f"Dataset {'duration'}: {f['duration'][()]}")
        print(f"Shape of dataset {'label'}: {f['label'].shape}")
    print("================================================================================")


"""
    ================================================================================
    Keys in the file: ['amp', 'label', 'pha']
    Shape of dataset amp: (4250, 3, 3, 30)
    Shape of dataset pha: (4250, 3, 3, 30)
    Shape of dataset label: (10, 4)
    ================================================================================
    Keys in the file: ['amp', 'label', 'pha']
    Shape of dataset amp: (3500, 3, 3, 30)
    Shape of dataset pha: (3500, 3, 3, 30)
    Shape of dataset label: (10, 4)
    ================================================================================
    Keys in the file: ['data', 'duration', 'label']
    Shape of dataset data: (5, 4250, 6)
    Dataset duration: 85
    Shape of dataset label: (10, 4)
    ================================================================================
    Keys in the file: ['data', 'duration', 'label']
    Shape of dataset data: (5, 3500, 6)
    Dataset duration: 70
    Shape of dataset label: (10, 4)
    ================================================================================
"""