import json
import os
import numpy as np
from tqdm import tqdm
import random
import re

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from openai import OpenAI


random.seed(42)
start = 0
PER_CLASS_SAMPLES = 100


# client = OpenAI()
client = OpenAI(
  base_url = "http://localhost:8000/v1/",
  api_key = "EMPTY"
)



unique_classes_dict = {'standing': 1, 'sitting':3, 'walking':5, 'run':7, 'bike':8}
unique_classes =[]
for key in unique_classes_dict.keys():
    unique_classes.append(key)

unique_classes = set(unique_classes)
print(unique_classes)

UNCLEAR_LABEL = "Unclear"
unique_classes_dict[UNCLEAR_LABEL] = 9

print(unique_classes_dict)



data_file_name = "/home/simran/SHL/data_20Hz_shl.npy"
label_file_name = "/home/simran/SHL/label_20Hz_shl.npy"


answer_format = "The identified class is: "
prompt = (
    "The given IMU sensor data can be associated with one of the following classes: "
    f"{unique_classes}. "
    "Write only the name of the identified class in the format "
    f"'{answer_format}'"
)


result_f = open("/home/simran/SLU/results/vicuna13b_shl.txt","w")


def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])


def gpt_imu_classification(accl_str, gyro_str):
    messages = [
        {
            "role": "system",
            "content": (
                "You're an expert on classifying human activity from IMU sensor data. "
                "The accelerometer data is normalized by being divided by 9.8."
            )
        },
        {
            "role": "user",
            "content": (
                f"{prompt}\n"
                f"Gyroscope: {gyro_str}\n"
                f"Accelerometer: {accl_str}"
            ),
        },
    ]
    params = {
        # "model": "ft:gpt-3.5-turbo-0125:worcester-polytechnic-institute::9WApCtou",
        "model": "vicuna-13b-v1.5-16k",
        "messages": messages,
        "max_tokens": 30,
    }

    result = client.chat.completions.create(**params)
    outputs = result.choices[0].message.content
    return outputs
label_count_dict = {}
for val in unique_classes_dict.keys():
    label_count_dict[val] = 0
print(label_count_dict)
final_count = 0
pred = []
gt = []

    
data = np.load(data_file_name)
label = np.load(label_file_name)
label_dict = {v: k for k, v in unique_classes_dict.items()}
print(label_dict)

for idx in tqdm(range(len(data))):
    # sample_index = random.randint(0, len(data))
    sample_index = idx
    sample_label = label_dict[int(label[sample_index])]
    if label_count_dict[sample_label] >=start and label_count_dict[sample_label] < start + PER_CLASS_SAMPLES:
        final_count +=1
        label_count_dict[sample_label]+=1
    else:
        label_count_dict[sample_label]+=1
        continue
    
    accl, gyro = data[sample_index][:, 0:3], data[sample_index][:, 3:6]
    accl = accl/9.8
    accl, gyro = accl.tolist(), gyro.tolist()
    accl_str = sensor_subsampled_string(accl)
    gyro_str = sensor_subsampled_string(gyro)

    result_f.write(f"True: {sample_label}\n")
    # outputs = eval_model(args)
    
    outputs = gpt_imu_classification(accl_str, gyro_str)
    result_f.write(f"{outputs}\n")
    
    gt.append(unique_classes_dict[sample_label])
    if answer_format not in outputs:
        result_f.write("INSTRUCTION AVOIDED\n")
        pred.append(unique_classes_dict[UNCLEAR_LABEL])
    else:
        pred_label = outputs.split(answer_format)[-1]
        pred_label = re.sub(r'[^\w\s]','', pred_label)
        pred_label = pred_label.strip()
        if pred_label.lower() in unique_classes_dict.keys():
            pred.append(unique_classes_dict[pred_label.lower()])
        else:
            result_f.write("CLASS OUT OF BOX\n")
            pred.append(unique_classes_dict[UNCLEAR_LABEL])

result_f.write(classification_report(
    gt, pred)
)   
result_f.close()
print(label_count_dict)
print(final_count)
print(label_count_dict)
cm = confusion_matrix(gt, pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=label_count_dict.keys()
)
disp.plot()
plt.savefig("/home/simran/SLU/results/shl_gpt_nemotron_non_f"+"_cm.png")
