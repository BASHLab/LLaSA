import json
import os
import numpy as np
from tqdm import tqdm
import random
import re

import torch

random.seed(42)
np.random.seed(42)


PER_CLASS_SAMPLES = 10


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
unique_classes_dict[UNCLEAR_LABEL] = idx+1

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
    # "First identify key features necessary to classify the given data "
    # "in a single sentence the format 'Key features are: '. Then, "
    "Write only the name of the identified class in the format "
    f"'{answer_format}'"
    # "Analyze the sensor data in 10 words and then provide the name of the "
    # "identified class as a summary followed by 'Class:'."
    # "Don't write a full sentence."
)

qa_sample_list = []

for dataset in datasets:
    data = np.load(os.path.join(root_data_dir, dataset, data_file_name))
    label = np.load(os.path.join(root_data_dir, dataset, label_file_name))
    label_dict = dataset_activity_label_dict[dataset]
    label_count_dict = {}
    for val in label_dict.values():
        label_count_dict[val] = 0
    last_axis = last_axis_dict[dataset]
    
    for idx in tqdm(range(len(data))):
        sample_index = random.randint(0, len(data))
        # sample_index = idx
        image_file = f"/hdd/LLM/limuBERT_data/train_llasa/images/{dataset}/sensor_image_{sample_index}.npy"
        if not os.path.exists(image_file):
            continue
        
        sample_label = label_dict[str(int(label[sample_index, 0, last_axis]))]
        if label_count_dict[sample_label] < PER_CLASS_SAMPLES:
            label_count_dict[sample_label]+=1
        else:
            continue
        
        qa_sample = {
            "id": f"{dataset}_{sample_label}_{sample_index}",
            "image": image_file,
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": answer_format+sample_label},
            ],
        }
        qa_sample_list.append(qa_sample)
    
with open(os.path.join("/hdd/LLM/SLU/limubert_classification.json"), "w") as outfile:
    json.dump(qa_sample_list, outfile)
