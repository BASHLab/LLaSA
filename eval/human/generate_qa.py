from pathlib import Path
import pandas as pd
import os
import re
import numpy as np
import json
from tqdm import tqdm
import random

from openai import OpenAI

random.seed(42)


client = OpenAI()

MAX_PER_CLASS = 2

sample_len = 600
data_dir = "/home/simran/SLU/eval/human/bash_qa_base"
location = "in the smartphone that the subject is holding with one hand"
categories = ["narration", "reliability"]
benchmark_dir = "/home/simran/SLU/eval/human/bash_qa_benchmark"

label_name_dict = {
    "swirl": "rotating",
    "sit": "sitting",
    "stand": "standing",
    "pick": "picking up objects from the floor",
    "object": "picking up objects from the floor",
    "jump": "jumping",
    "rotat": "rotating",
    "spin": "rotating",
    "stretch": "stretching or light exercises",
    "exercise": "stretching or light exercises",
    "light": "stretching or light exercises",
    "stair": "using stairs",
    "elevator": "using elevator",
    "walk": "walking"
}


error_log = open("/home/simran/SLU/eval/human/error.log","w")


def get_qa_pairs(output):
    lines = output.split("\n")
    
    qa_pairs = []
    for line in lines:
        if "Q:" in line:
            q = line.split("Q:")[-1].strip()
        elif "A:" in line:
            a = line.split("A:")[-1].strip()
            qa_pairs.append((q,a))
    
    assert len(qa_pairs)==3
    
    return qa_pairs


def openended_qa_generation(npy_str, label, category):
    
    if category == "science":
        prompt = (
            "You're an expert on the scientific knowledge behind analyzing "
            f"IMU sensor data. The IMU sensors are located {location}. "
            "Based on the data and given summary, please generate "
            "3 question-answer pairs that require scientific reasoning. "
        )
    elif category == "narration":
        prompt = (
            "You're an expert at narrating IMU sensor data for human activities. "
            f"The IMU sensors are located {location}. "
            "Based on the data and given summary, please generate "
            "3 question-answer pairs that related to what "
            "is happening that require expert analysis of the data "
            "and knowledge about the correlation of the activity "
            "and the sensor data."
        )
    elif category == "reliability":
        prompt = (
            "You're an expert at analyzing the reliability of collected IMU data. "
            f"The IMU sensors are located {location}. "
            "Based on the data and given summary, please generate "
            "3 question-answer pairs that require you to "
            "analyze the reliability of the given IMU data in the appropriate "
            "human activity context."
        )
    else:
        raise ValueError(f"{category} is an invalid category")

    messages = [
        {
            "role": "system",
            "content": (
                f"{prompt}"
                "Write the QA pairs in the following format: "
                "1. Q: \n A: \n 2. Q: \n A: \n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Summary: {label}\n"
                f"{npy_str}"
            ),
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 2000,
    }

    result = client.chat.completions.create(**params)
    outputs = result.choices[0].message.content
    return outputs

def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])


def subsample_data(path, n=60):
    for label_key in label_name_dict:
        if label_key in str(path).lower():
            label_name = label_name_dict[label_key]
            break
    
    try:
        user = re.findall(r"u\d+_", str(path).lower())[0]
    except:
        print(path)
    
    accl_df = pd.read_csv(path)
    gyro_df = pd.read_csv(str(path).replace("Accelerometer","Gyroscope"))
    
    df = pd.concat(
        [accl_df[["x","y","z"]], gyro_df[["x","y","z"]]], axis=1
    )
    activity_data = df.to_numpy()
    npy_name_base = f"{user}{label_key}"
    
    npy_file_to_string = {}
    
    for npy_idx, start_idx in enumerate(
        range(
            201, len(activity_data)-200, sample_len
        )
    ):
        data_chunk = activity_data[start_idx:start_idx+sample_len]
        indices_20hz = np.round(np.linspace(0, len(data_chunk) - 1, 120)).astype(int)
        data_chunk_subsampled = data_chunk[indices_20hz]
        npy_name = npy_name_base + f"_{npy_idx}.npy"
        # np.save(os.path.join(data_dir, npy_name), data_chunk_subsampled)
        
        accl = data_chunk_subsampled[:,0:3]
        gyro = data_chunk_subsampled[:,0:6]
        accl, gyro = accl.tolist(), gyro.tolist()
        accl_str = sensor_subsampled_string(accl)
        gyro_str = sensor_subsampled_string(gyro)
        
        npy_file_to_string[npy_name] = (
            f"Accelerometer: {accl_str} Gyroscope: {gyro_str}"
        )
    
    subset = dict(random.choices(list(npy_file_to_string.items()),k=2))
    
    return subset


npy_file_to_string_all = {}
num_user_class=0
for path in Path('/home/simran/SLU/eval/human/llasa_data').rglob('*Accelerometer.csv'):
    npy_file_to_string = subsample_data(path)
    npy_file_to_string_all.update(npy_file_to_string)
    num_user_class+=1
# print(num_user_class)

dropped=0

for category in categories:
    print(category)
    count_samples=0
    qa_sample_list = []
    for npy_filepath, npy_str in tqdm(npy_file_to_string_all.items()):
        for label_key in label_name_dict:
            if label_key in npy_filepath:
                label = label_name_dict[label_key]
                break
        
        outputs = openended_qa_generation(
            npy_str=npy_str, label=label, category=category
        )
        # outputs = ""
        print(outputs)
        count_samples+=1
        # if count_samples > 1:
        #     break
        try:
            qa_pairs = get_qa_pairs(outputs)
            for qa_idx, (q, a) in enumerate(qa_pairs):
                qa_sample = {
                    "id": npy_filepath.split("/")[-1].split(".")[0],
                    "image": npy_filepath,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{q}"},
                        {"from": "gpt", "value": a},
                    ],
                }
                qa_sample_list.append(qa_sample)
        except Exception as e:
            error_log.write(npy_filepath)
            error_log.write(f" {category} Error: {e}\n")
            dropped += 1
    with open(os.path.join(benchmark_dir,f"{category}_test.jsonl"), "w") as outfile:
        json.dump(qa_sample_list, outfile)

print(f"Dropped: {dropped}")