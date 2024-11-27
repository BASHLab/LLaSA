import random
import os
import json

import numpy as np
from tqdm import tqdm

from openai import OpenAI

random.seed(100)
client = OpenAI()


MAX_PER_CLASS = 2

categories = ["math"] # ["science", "narration", "reliability", "math"]

benchmark_dir = "/hdd/LLM/PAMAP2/qa_benchmark"

label_dict = {
    '1': "lying",
    '2': "sitting",
    '3': "standing",
    '4': "walking",
    '5': "running",
    '6': "cycling",
    '7': "Nordic walking",
    '9': "watching TV",
    '10': "computer work",
    '11': "car driving",
    '12': "ascending stairs",
    '13': "descending stairs",
    '16': "vacuum cleaning",
    '17': "ironing",
    '18': "folding laundry",
    '19': "house cleaning",
    '20': "playing soccer",
    '24': "rope jumping"
}


def get_qa_pairs(output):
    lines = output.split("\n")
    
    qa_pairs = []
    for line in lines:
        if "Q:" in line:
            q = line.split("Q:")[-1].strip()
        elif "A:" in line:
            a = line.split("A:")[-1].strip()
            qa_pairs.append((q,a))
    
    assert len(qa_pairs)==5
    
    return qa_pairs


def openended_qa_generation(accl_str, gyro_str, label, location, category):
    if category == "science":
        prompt = (
            "You're an expert on the scientific knowledge behind analyzing "
            f"IMU sensor data. The IMU sensors are located at {location}. "
            "Based on the data and given summary, please generate "
            "5 question-answer pairs that require scientific reasoning. "
        )
    elif category == "narration":
        prompt = (
            "You're an expert at narrating IMU sensor data for human activities. "
            f"The IMU sensors are located at {location}. "
            "Based on the data and given summary, please generate "
            "5 question-answer pairs that related to what "
            "is happening that require expert analysis of the data "
            "and knowledge about the correlation of the activity "
            "and the sensor data."
        )
    elif category == "reliability":
        prompt = (
            "You're an expert at analyzing the reliability of collected IMU data. "
            f"The IMU sensors are located at {location}. "
            "Based on the data and given summary, please generate "
            "5 question-answer pairs that require you to "
            "analyze the reliability of the given IMU data in the appropriate "
            "human activity context."
        )
    elif category == "math":
        prompt = (
            "You're an expert at solving IMU and human activity related mathematical problems. "
            f"The IMU sensors are located at {location}. "
            "Based on the data and given summary, please generate "
            "5 question-answer pairs where the question is a small relevant "
            "mathematical problem and the answer is a very short but detailed solution. "
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
                f"Gyroscope: {gyro_str}\n"
                f"Accelerometer: {accl_str}"
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


sensor_locations = ['ankle', 'chest', 'hand']

error_log = open(os.path.join(benchmark_dir,"error.log"),"w")

dropped = 0
for category in categories:
    qa_sample_list = []
    for subject_id in range(1,9):
        label = np.load(f'/hdd/LLM/PAMAP2/original/subject10{subject_id}.dat/label.npy')
        for sensor_location in sensor_locations:
            count_dict = {v: 0 for v in label_dict.values()}
            
            data = np.load(f'/hdd/LLM/PAMAP2/original/subject10{subject_id}.dat/{sensor_location}.npy')

            while sum(count_dict.values()) < MAX_PER_CLASS*len(label_dict):
                sample_index = random.randint(0, len(data)-1)
                if label[sample_index]=='0':
                    continue
                else:
                    label_name = label_dict[label[sample_index]]
                    count_dict[label_name]+=1
                    if count_dict[label_name] > MAX_PER_CLASS:
                        continue
                
                accl, gyro = data[sample_index][:, 0:3], data[sample_index][:, 3:6]
                accl, gyro = accl.tolist(), gyro.tolist()
                accl_str = sensor_subsampled_string(accl)
                gyro_str = sensor_subsampled_string(gyro)
                
                outputs = openended_qa_generation(
                    accl_str=accl_str, gyro_str=gyro_str,
                    label=label_name, location=sensor_location,
                    category=category
                )
                
                print(outputs)
                
                sample_id = f"{sensor_location}_{label_name}_{sample_index}"
                npy_filepath = os.path.join(
                    benchmark_dir,
                    f"data/{sample_id}.npy"
                )
                
                np.save(npy_filepath, data[sample_index])
                
                try:
                    qa_pairs = get_qa_pairs(outputs)
                    for qa_idx, (q, a) in enumerate(qa_pairs):
                        qa_sample = {
                            "id": f"{sample_id}_{qa_idx}",
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