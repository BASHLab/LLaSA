import numpy as np


def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])

filepath = "/hdd/LLM/PAMAP2/qa_benchmark/data/hand_ironing_195.npy"

data = np.load(filepath)

accl, gyro = data[:, 0:3], data[:, 3:6]
accl, gyro = accl.tolist(), gyro.tolist()
accl_str = sensor_subsampled_string(accl)
gyro_str = sensor_subsampled_string(gyro)

with open("/hdd/LLM/SLU/acclgyro.txt","w") as f:
    f.write(f"Accelerometer: {accl_str}\nGyroscope: {gyro_str}")