import numpy as np 
from matplotlib import pyplot as plt


fontsize=20

with open("/home/simran/SLU/eval/human/filelist.txt","r") as f:
    files=f.readlines()


def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])

with open("/home/simran/SLU/results/openended/assessments_5_13b.txt","r") as f:
    assessments = f.readlines()


f_prompt = open("/home/simran/SLU/eval/human/prompts.txt","w")
for idx,filename in enumerate(files):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (14, 14))
    
    data = np.load(filename.strip())
    
    accl, gyro = data[:, 0:3], data[:, 3:6]
    accl, gyro = accl.tolist(), gyro.tolist()
    accl_str = sensor_subsampled_string(accl)
    gyro_str = sensor_subsampled_string(gyro)

    for q_idx, line in enumerate(assessments):
        if filename in line:
            q = assessments[q_idx+1].strip()
            break
    f_prompt.write(f"{q} Accelerometer: {accl_str} Gyroscope: {gyro_str}\n")
    
    accl_x = data[:,0]
    accl_y = data[:,1]
    accl_z = data[:,2]
    
    gyro_x = data[:,3]
    gyro_y = data[:,4]
    gyro_z = data[:,5]
    
    ax[0].plot(np.linspace(0, 6, 120), accl_x, label = 'X-axis readings', color = '#e41a1c')
    ax[0].plot(np.linspace(0, 6, 120), accl_y, label = 'Y-axis readings', color = '#377eb8')
    ax[0].plot(np.linspace(0, 6, 120), accl_z, label = 'Z-axis readings', color = '#4daf4a')
    
    ax[1].plot(np.linspace(0, 6, 120), gyro_x, label = 'X-axis readings', color = '#e41a1c')
    ax[1].plot(np.linspace(0, 6, 120), gyro_y, label = 'Y-axis readings', color = '#377eb8')
    ax[1].plot(np.linspace(0, 6, 120), gyro_z, label = 'Z-axis readings', color = '#4daf4a')
    
    ax[0].set_title ("Accelerometer Data", fontsize=fontsize*1.5)
    ax[0].set_xlabel("time $[s]$", fontsize=fontsize)
    ax[0].set_ylabel("value $[m/s^2]$", fontsize=fontsize)
    # ax[0].legend(fontsize=fontsize)
    
    ax[1].set_title ("Gyroscope Data", fontsize=fontsize*1.5)
    ax[1].set_xlabel("time $[s]$", fontsize=fontsize)
    ax[1].set_ylabel("value $[deg/s]$", fontsize=fontsize)
    # ax[1].legend(fontsize=fontsize)
    
    ax[0].tick_params(labelsize=fontsize)
    ax[0].tick_params(labelsize=fontsize)
    
    ax[1].tick_params(labelsize=fontsize)
    ax[1].tick_params(labelsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(f"/home/simran/SLU/eval/human/{idx}.png", pad_inches=5)
