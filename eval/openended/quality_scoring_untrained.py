import random
import os
import json
import csv

import numpy as np
from tqdm import tqdm

from openai import OpenAI

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import torch

random.seed(190)
np.random.seed(190)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


client = OpenAI()


categories = ["science", "narration", "reliability"]  #, "math"]

model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-lora"


def get_gpt_answer(accl_str, gyro_str, q):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant."
            )
        },
        {
            "role": "user",
            "content": (
                f"{q}\n"
                f"Gyroscope: {gyro_str}\n"
                f"Accelerometer: {accl_str}"
            ),
        },
    ]
    params = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 250,
    }

    result = client.chat.completions.create(**params)
    outputs = result.choices[0].message.content
    return outputs


def quality_scoring(q, standard_answer, activity_label, sensor_location, predicted_answer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at assessing the quality of predicted answers "
                "based on questions related to IMU data and given information. "
                "Please use the 'Standard Answer', 'Activity Label', and 'Sensor Location' "
                "to assess the 'Predicted Answer' "
                "based on the relevance " # correctness, completeness, consistency, and helpfulness "
                "of the predicted answer. "
                # "Give only a single overall score in the following "
                # "format on a scale of 0 to 100:\n"
                # "Quality score: "
                "In a short sentence explain why the answer is good or bad. "
                "Assessment: "
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {q}\n"
                f"Standard Answer: {standard_answer}\n"
                f"Activity Label: {activity_label}\n"
                f"Sensor Location: {sensor_location}\n"
                f"Predicted Answer: {predicted_answer}"
            ),
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 200,
    }

    result = client.chat.completions.create(**params)
    outputs = result.choices[0].message.content
    return outputs
    


def sensor_subsampled_string(data, n=60):
    if len(data)/n>10:
        print(f"High compression: {len(data)/n}")
    indices = np.round(np.linspace(0, len(data) - 1, n)).astype(int)
    return str([list(np.round(data[idx],6)) for idx in indices])

fields = [
    "Q", "Standard Answer (GPT-4o)", "Activity Label", "Sensor Location", "IMU filepath",
    "GPT-3.5-Turbo answer", "LLaSA answer", "GPT-3.5-T quality score", "LLaSA quality score"
]

result_summary_f = open('/hdd/LLM/SLU/results/openended/summary.txt','w')
assessments_f = open('/hdd/LLM/SLU/results/openended/assessments.txt','w')

for category in categories:
    total_gpt_score = 0
    total_llasa_score = 0
    total_numerical_scores = 0
    
    # csvfile = open(f"/hdd/LLM/SLU/results/openended/{category}_results.csv", 'w')
    # csvwriter = csv.writer(csvfile)
    # csvwriter.writerow(fields)
    
    category_test_data = json.load(open(f'/hdd/LLM/PAMAP2/qa_benchmark/{category}_test.jsonl', 'r'))
    category_test_data = random.sample(category_test_data, 15)
    for sample in tqdm(category_test_data):
        sensor_location = sample["id"].split("_")[0]
        activity_label = sample["id"].split("_")[1]
        filepath = sample["image"]
        q = sample["conversations"][0]["value"].replace("<image>\n", "")
        standard_answer = sample["conversations"][1]["value"]
        
        data = np.load(filepath)
        
        # accl, gyro = data[:, 0:3], data[:, 3:6]
        # accl, gyro = accl.tolist(), gyro.tolist()
        # accl_str = sensor_subsampled_string(accl)
        # gyro_str = sensor_subsampled_string(gyro)

        # gpt_answer = get_gpt_answer(accl_str, gyro_str, q)
        # gpt_score_explanation = quality_scoring(
        #     q, standard_answer, activity_label, sensor_location, gpt_answer
        # )
        # print(gpt_score_explanation)
        
        # if gpt_score_explanation.split()[2].isnumeric():
        #     gpt_score = int(gpt_score_explanation.split()[2])
        # else:
        #     print(f"Nonnumerical score")
        
        data[:,0:3] = data[:,0:3]/9.8
        temp_filepath = "/hdd/LLM/SLU/temp.npy"
        np.save(temp_filepath, data)
        
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": "lmsys/vicuna-7b-v1.5",
            "model_name": get_model_name_from_path(model_path),
            "query": q,
            "conv_mode": None,
            "image_file": temp_filepath,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 500
        })()
        llasa_answer = eval_model(args)
        llasa_score_explanation = quality_scoring(
            q, standard_answer, activity_label, sensor_location, llasa_answer
        )
        print(llasa_score_explanation)
        
        assessments_f.write(filepath+"\n"+q+"\n"+llasa_answer+"\n")
        assessments_f.write(llasa_score_explanation+"\n-----------------------------------\n")
        # if llasa_score_explanation.split()[2].isnumeric():
        #     llasa_score = int(llasa_score_explanation.split()[2])
        # else:
        #     print(f"Nonnumerical score")
        
        
    #     csv_row = [
    #         q, standard_answer, activity_label, sensor_location, filepath,
    #         gpt_answer, llasa_answer, gpt_score_explanation, llasa_score_explanation
    #     ]
        
    #     csvwriter.writerow(csv_row)
        
    #     if isinstance(gpt_score, int) and isinstance(llasa_score, int):
    #         total_gpt_score += gpt_score
    #         total_llasa_score += llasa_score
    #         total_numerical_scores += 1
    
    # result_summary_f.write(
    #     f"{category} ({total_numerical_scores}) GPT: {total_gpt_score/total_numerical_scores} "
    #     f"LLaSA: {total_llasa_score/total_numerical_scores}"
    # )
