from ast import literal_eval
import os
import json

import numpy as np
from tqdm import tqdm

from openai import OpenAI


client = OpenAI()


dataset = "pamap2"
sensorcaps_path = f"/hdd/LLM/sensorcaps_10hz_{dataset}_v2.jsonl"


def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])


def moving_average(a, n=12):
    ma = np.zeros((a.shape[0]-n+1, a.shape[1]), dtype=float)
    for i in range(a.shape[1]):
        ma[:,i] = np.convolve(a[:,i], np.ones(n), 'valid') / n
    return ma

def round_str(a):
    return str(np.round(a, 6))


def limubert_sample_to_sensorcaps(summary, accl, gyro, location):
    accl_str = sensor_subsampled_string(accl)
    gyro_str = sensor_subsampled_string(gyro)
    
    accl_np = np.array(accl)
    gyro_np = np.array(gyro)
    
    accl_ma = moving_average(accl_np)
    gyro_ma = moving_average(gyro_np)
    
    accl_fft = np.fft.fft(accl_np)
    gyro_fft = np.fft.fft(gyro_np)
    
    accl_min = np.min(accl_np, axis=-1)
    gyro_min = np.min(gyro_np, axis=-1)
    accl_max = np.max(accl_np, axis=-1)
    gyro_max = np.max(gyro_np, axis=-1)
    
    accl_med = np.median(accl_np, axis=-1)
    gyro_med = np.median(gyro_np, axis=-1)
    accl_var = np.var(accl_np, axis=-1)
    gyro_var = np.var(gyro_np, axis=-1)
    
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
                "Additional information computed for each axis is provided afterwards, "
                "with the windows for moving averages being 12. "
                "Finally, the sensor location is provided after 'Location:'. "
                "You should provide a comprehensive details of at least a couple of characteristic IMU "
                "features for that event would be within 20-25 words, followed by 'Features:'. "
                "The IMU features should be concise and descriptive. Separate multiple features "
                "with commas. Derive the list of features based on the given sensor data. "
                "Then, narrate the temporal event with details that are context-aware "
                "based on the sensor data, followed by 'Narration:', in a step-by-step "
                "fashion, analyzing it within 500 words or less. "
                "Please analyze even the small movements thoroughly in a logical and "
                "contextual manner, utilizing deductive thought process while "
                "being aware of the knowledge regarding the meaning of the "
                "sensor data. Use descriptive terms that aren't vague."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summary: {summary}\n"
                f"Gyroscope: {gyro_str}\n"
                f"Accelerometer: {accl_str}\n"
                f"Gyroscope moving average: {round_str(gyro_ma)}\n"
                f"Accelerometer moving average: {round_str(accl_ma)}\n"
                f"Gyroscope FFT: {round_str(gyro_fft)}\n"
                f"Accelerometer FFT: {round_str(accl_fft)}\n"
                f"Gyroscope minimum: {round_str(gyro_min)}\n"
                f"Accelerometer minimum: {round_str(accl_min)}\n"
                f"Gyroscope maximum: {round_str(gyro_max)}\n"
                f"Accelerometer maximum: {round_str(accl_max)}\n"
                f"Gyroscope median: {round_str(gyro_med)}\n"
                f"Accelerometer median: {round_str(accl_med)}\n"
                f"Gyroscope variance: {round_str(gyro_var)}\n"
                f"Accelerometer variance: {round_str(accl_var)}\n"
                f"Location: {location}"
            ),
        },
    ]
    params = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1500,
    }

    result = client.chat.completions.create(**params)
    narration = result.choices[0].message.content
    messages.append({"role": "assistant", "content": narration})
    sensorcaps_sample = json.dumps({"messages": messages})
    
    
    
    return sensorcaps_sample


root_data_dir = "/hdd/LLM/PAMAP2"


with open('/hdd/LLM/limuBERT_data/dataset_activity_label.json') as json_file:
    dataset_activity_label_dict = json.load(json_file)

label_dict = dataset_activity_label_dict[dataset.upper()]

sensor_locations = ['ankle', 'chest', 'hand']

with open(sensorcaps_path,"w") as f:
    for subject_id in range(1,9):
        label = np.load(f'/hdd/LLM/PAMAP2/original/subject10{subject_id}.dat/label.npy')
        for location in sensor_locations:
            data = np.load(f'/hdd/LLM/PAMAP2/original/subject10{subject_id}.dat/{location}.npy')


            for sample_index in tqdm(range(len(data))):
                sample_data = data[sample_index]
                if label[sample_index]=='0':
                    continue
                else:
                    sample_label = label_dict[label[sample_index]]
                accl, gyro = sample_data[:, 0:3], sample_data[:, 3:6]
                accl, gyro = accl.tolist(), gyro.tolist()
                
                sensorcaps_sample = limubert_sample_to_sensorcaps(
                    summary=sample_label,
                    accl=accl,
                    gyro=gyro,
                    location=location
                )
                f.write(sensorcaps_sample+"\n")


def sensorcaps_to_opensqa(qa_context):
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
                "Please generate 5 detailed question-answer pairs that require "
                "step-by-step logical deductive thinking and knowledgable analysis of the "
                "sensor data. Please make the questions complex but logical so that they "
                "require information that can be derived based on the vast knowledge about "
                "sensor data and IMU activities. The questions and answers need to "
                "acknowledge the context given by the user. Please write them in "
                "the following list format: 1. Q: Question \n A: Answer \n"
            )
        },
        {
            "role": "user",
            "content": (qa_context),
        },
    ]
    params = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 3000,
    }

    result = client.chat.completions.create(**params)
    qas = result.choices[0].message.content
    messages.append({"role": "assistant", "content": qas})
    opensqa_sample = json.dumps({"messages": messages})
    return qas, opensqa_sample


with open(sensorcaps_path, "r") as f:
    samples = f.readlines()


opensqa_path = f"/hdd/LLM/opensqa_10hz_{dataset}_v2.jsonl"
with open(opensqa_path,"w") as f:
    for sample in tqdm(samples):
        sample = literal_eval(sample)['messages']
        sensor_and_summary = sample[1]['content']
        feature_and_narration = sample[2]['content']
        qa_context = sensor_and_summary + "\n" + feature_and_narration
        _, opensqa_sample = sensorcaps_to_opensqa(qa_context)
        f.write(opensqa_sample+"\n")
