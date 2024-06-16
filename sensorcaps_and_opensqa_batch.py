from ast import literal_eval
import os
import json
from time import sleep

import numpy as np
from tqdm import tqdm

from openai import OpenAI

client = OpenAI()


sensorcaps_path = "sensorcaps_10hz_full.jsonl"
opensqa_dir = "opensqa_10hz_requests"

WAITING_STATUSES = [
    "validating",
    "in_progress",
    "finalizing"
]

done_batches = 0

def chunks(lst, n=1000):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])


def limubert_sample_to_sensorcaps(summary, accl_str, gyro_str):
    messages = [
        {
            "role": "system",
            "content": (
                "Please consider yourself to be an expert on gyroscope and accelerometer sensor "
                "information given as a metadata of IMU datasets."
                "You are given the IMU sensor readings of a human activity. "
                "The user also provides a brief summary of the event followed by 'Summary:'. "
                "They also give you gyroscopic and accelerometer sensor data followed by "
                "'Gyroscope:' and 'Accelerometer:' respectively. "
                "They are written in a Python list of lists format and contain x, y, and z "
                "axis data respectively. Please pay attention to the values and the signs. "
                "You should provide a comprehensive details of at least a couple of characteristic IMU "
                "features for that event would be within 10-15 words, followed by 'Features:'. "
                "The IMU features should be concise and descriptive. Separate multiple features "
                "with commas. Derive the list of features based on the given sensor data. "
                "Then, narrate the temporal event with details that are context-aware "
                "based on the sensor data, followed by 'Narration:', in a step-by-step "
                "fashion, analyzing it within 150 words or less. "
                "Although the movements might appear to be minimal, please "
                "analyze the small movements thoroughly in a logical and "
                "contextual manner, utilizing deductive thought process while "
                "being aware of the knowledge regarding the meaning of the "
                "sensor data. Don't call the movements minimal or subtle. "
                "Instead use descriptive terms that aren't vague."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summary: {summary}\n"
                f"Gyroscope: {gyro_str}\n"
                f"Accelerometer: {accl_str}"
            ),
        },
    ]
    params = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 300,
    }

    result = client.chat.completions.create(**params)
    narration = result.choices[0].message.content
    messages.append({"role": "assistant", "content": narration})
    sensorcaps_sample = json.dumps({"messages": messages})
    return sensorcaps_sample


root_data_dir = "/hdd/LLM/limuBERT_data/extracted_data"
datasets = sorted(os.listdir(root_data_dir))

data_file_name = "data_20_120.npy"
label_file_name = "label_20_120.npy"


with open('/hdd/LLM/limuBERT_data/dataset_activity_label.json') as json_file:
    dataset_activity_label_dict = json.load(json_file)


last_axis_dict = {
    "hhar": 2, "motion":0, "shoaib":0, "uci":0
}
# with open(sensorcaps_path,"w") as f:
#     for dataset in datasets:
#         data = np.load(os.path.join(root_data_dir, dataset, data_file_name))
#         label = np.load(os.path.join(root_data_dir, dataset, label_file_name))
#         label_dict = dataset_activity_label_dict[dataset]
#         last_axis = last_axis_dict[dataset]
#         print(dataset)
        
        
#         for sample_index in tqdm(range(len(data))):
#             sample_data = data[sample_index]
#             key = str(int(label[sample_index, 0, last_axis]))
#             sample_label = label_dict[str(int(label[sample_index, 0, last_axis]))]
#             accl, gyro = data[0][:, 0:3], data[0][:, 3:6]
#             accl, gyro = accl.tolist(), gyro.tolist()
#             accl_str = sensor_subsampled_string(accl)
#             gyro_str = sensor_subsampled_string(gyro)
#             sensorcaps_sample = limubert_sample_to_sensorcaps(
#                 summary=sample_label,
#                 accl_str=accl_str,
#                 gyro_str=gyro_str
#             )
#             f.write(sensorcaps_sample+"\n")


def sensorcaps_to_opensqa_requests(qa_context, custom_id):
    messages = [
        {
            "role": "system",
            "content": (
                "Please consider yourself to be an expert on gyroscope and accelerometer sensor "
                "information of IMU datasets."
                "You are given the IMU sensor readings of a human activity. "
                "The user also provides a brief summary of the event followed by 'Summary:'. "
                "They also give you gyroscopic and accelerometer sensor data followed by "
                "'Gyroscope:' and 'Accelerometer:' respectively. "
                "Characteristic IMU features of the event are written after 'Features:'. "
                "Finally, there's a temporal narration of the events after 'Narration:'. "
                "Please generate 10 detailed question-answer pairs that require "
                "step-by-step logical deductive thinking and knowledgable analysis of the "
                "sensor data. Please make the questions complex but logical so that they "
                "require information that can be derived based on the vast knowledge about "
                "sensor data and IMU activities. The questions and answers need to "
                "acknowledge the context given by the user."
            )
        },
        {
            "role": "user",
            "content": (qa_context),
        },
    ]
    params = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 1000,
    }
    
    request_format = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": params
    }

    return json.dumps(request_format)


with open(sensorcaps_path, "r") as f:
    samples = f.readlines()

sample_batches = list(chunks(samples, n=200))
for idx, sample_batch in tqdm(enumerate(sample_batches)):
    if idx < done_batches:
        continue
    batch_file_name = os.path.join(opensqa_dir, str(idx)+".jsonl")
    with open(batch_file_name, "w") as f:
        for sample_idx, sample in enumerate(sample_batch):
            sample = literal_eval(sample)['messages']
            sensor_and_summary = sample[1]['content']
            feature_and_narration = sample[2]['content']
            qa_context = sensor_and_summary + "\n" + feature_and_narration
            opensqa_request = sensorcaps_to_opensqa_requests(
                qa_context,
                custom_id = "chunk_"+str(idx)+"_"+str(sample_idx)
            )
            f.write(opensqa_request+"\n")
    batch_input_file = client.files.create(
        file=open(batch_file_name, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_return = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"OpenSQA Chunk #{idx}"
        }
    )
    batch_id = batch_return.id
    status = client.batches.retrieve(batch_id).status
    while status in WAITING_STATUSES:
        sleep(10)
        status = client.batches.retrieve(batch_id).status
    if status == "failed":
        raise RuntimeError(f"Error with batch {batch_id}")
    
