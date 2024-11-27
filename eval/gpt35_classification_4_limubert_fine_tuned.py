import json
import os
import numpy as np
from tqdm import tqdm
import random
import re

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from openai import OpenAI


random.seed(42)
start = 150
PER_CLASS_SAMPLES = 100

client = OpenAI()

from openai import OpenAI
import json


client = OpenAI()


with open("/hdd/LLM/limuBERT_data/dataset_activity_label.json","r") as f:
    dataset_activity_label_dict = json.load(f)

unique_classes = set()
for val in dataset_activity_label_dict.values():
    unique_classes.update(val.values())
print(unique_classes)


unique_classes_dict = {}

for idx, unique_class in enumerate(unique_classes):
    unique_classes_dict[unique_class] = idx

UNCLEAR_LABEL = "Unclear"
unique_classes_dict[UNCLEAR_LABEL] = idx +1

print(unique_classes_dict)


root_data_dir = "/hdd/LLM/limuBERT_data/extracted_data"
datasets = sorted(os.listdir(root_data_dir))

data_file_name = "data_20_120.npy"
label_file_name = "label_20_120.npy"

last_axis_dict = {
    "hhar": 2, "motion":0, "shoaib":0, "uci":0
}

answer_format = "The identified class is: "
prompt = (
    "The given IMU sensor data can be associated with one of the following classes: "
    f"{unique_classes}. "
    "Write only the name of the identified class in the format "
    f"'{answer_format}'"
)


result_f = open("/hdd/LLM/SLU/simpletest_gpt35_nur_finetuned.txt","w")


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
        "model": "ft:gpt-3.5-turbo-0125:worcester-polytechnic-institute::9WApCtou",
        "messages": messages,
        "max_tokens": 30,
    }

    result = client.chat.completions.create(**params)
    outputs = result.choices[0].message.content
    return outputs
label_count_dict = {}
for val in unique_classes_dict.keys():
    label_count_dict[val] = 0
final_count = 0
pred = []
gt = []
for dataset in datasets:
    
    
    result_f.write(dataset+"\n")
    data = np.load(os.path.join(root_data_dir, dataset, data_file_name))
    label = np.load(os.path.join(root_data_dir, dataset, label_file_name))
    label_dict = dataset_activity_label_dict[dataset]
    
    last_axis = last_axis_dict[dataset]
    
    for idx in tqdm(range(len(data))):
        # sample_index = random.randint(0, len(data))
        sample_index = idx
        sample_label = label_dict[str(int(label[sample_index, 0, last_axis]))]
        if label_count_dict[sample_label] >=start and label_count_dict[sample_label] < start + PER_CLASS_SAMPLES:
            final_count +=1
            label_count_dict[sample_label]+=1
        else:
            label_count_dict[sample_label]+=1
            continue
        
        accl, gyro = data[sample_index][:, 0:3], data[sample_index][:, 3:6]
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
            if pred_label in unique_classes_dict.keys():
                pred.append(unique_classes_dict[pred_label])
            else:
                result_f.write("CLASS OUT OF BOX\n")
                pred.append(unique_classes_dict[UNCLEAR_LABEL])

result_f.write(classification_report(
        gt, pred, 
        labels=list(range(len(unique_classes_dict))),
        target_names=unique_classes_dict.keys())
    )   
result_f.close()
print(label_count_dict)
print(final_count)
