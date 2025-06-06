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


torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SAMPLE_PER_CATEGORY = 50


client = OpenAI()


categories = ["narration", "reliability"]  #,  "math"]
benchmark_dir = "/home/simran/SLU/eval/human/bash_qa_benchmark"
image_dir = "/home/simran/SLU/eval/human/bash_qa_base" 

# model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-lora"
# model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa_v2_small"
# model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa2-lr1e-4-mmlr1e-4-e1-r8-a8-lora"

checkpoint_dir = "/home/simran/LLaVA/checkpoints3/"
model_names = [
    "llava-v1.5-13b-llasa2-lr3e-5-mmlr1e-4-e1-r8-a8-proj1-lora"
    # dirname 
    # for dirname in os.listdir(checkpoint_dir) 
    # if dirname.startswith('llava-v1.5-13b-llasa2-lr')
]

print(model_names)

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
        "model": "gpt-4o-mini",
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
                "based on the correctness, completeness, consistency, and helpfulness "
                "of the predicted answer. "
                "Give a single overall score in the following "
                "format on a scale of 0 to 100:\n"
                "Quality score: \n"
                "Then give a short summary of the assessment in a couple of sentences "
                "followed by 'Assessment:'. "
                "Finally, provide a verdict whether the answer is good or bad followed by "
                "'Verdict:'."
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
        "max_tokens": 300,
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
    "GPT-4o-mini answer", "LLaSA answer", "GPT-4o-T quality score", "LLaSA quality score"
]

# result_summary_f = open('/home/simran/SLU/results/openended/summary.txt','w')


for model_name in model_names:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    
    average_score_all_categories = 0
    assessments_f = open(f'/home/simran/SLU/results/openended/assessments_human_{model_name}.txt','w')
    model_path = os.path.join(checkpoint_dir, model_name)
    for category in categories:
        total_gpt_score = 0
        total_llasa_score = 0
        total_numerical_scores = 0
        
        # csvfile = open(f"/home/simran/SLU/results/openended/{category}_results.csv", 'w')
        # csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(fields)
        print(os.path.join(benchmark_dir,f'{category}_test.jsonl'))
        print(os.path.exists(os.path.join(benchmark_dir,f'{category}_test.jsonl')))
        category_test_data = json.load(
            open(os.path.join(benchmark_dir,f'{category}_test.jsonl'),'r')
        )
        print(len(category_test_data))
        category_test_data = random.sample(category_test_data, SAMPLE_PER_CATEGORY)
        for sample in tqdm(category_test_data):
            sensor_location = sample["id"].split("_")[0]
            activity_label = sample["id"].split("_")[1]
            filepath = sample["image"]
            q = sample["conversations"][0]["value"].replace("<image>\n", "")
            standard_answer = sample["conversations"][1]["value"]
            
            data = np.load(os.path.join(image_dir,filepath))
            
            accl, gyro = data[:, 0:3], data[:, 3:6]
            accl, gyro = accl.tolist(), gyro.tolist()
            accl_str = sensor_subsampled_string(accl)
            gyro_str = sensor_subsampled_string(gyro)

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
            temp_filepath = "/home/simran/SLU/temp.npy"
            np.save(temp_filepath, data)
            
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": "lmsys/vicuna-7b-v1.5",
                # "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": q,
                # "query": f"Gyroscope: {gyro_str} Accelerometer: {accl_str}\n{q}",
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
            
            assessments_f.write(
                filepath+"\n"+q+"\nLLaSA: "+str(llasa_answer.encode("utf8"))
                +"\nStandard: "+standard_answer+"\n"
            )
            assessments_f.write(llasa_score_explanation+"\n-----------------------------------\n")
            if llasa_score_explanation.split()[2].isnumeric():
                llasa_score = int(llasa_score_explanation.split()[2])
            else:
                print(f"Nonnumerical score")
            
            
            if isinstance(llasa_score, int):
                total_llasa_score += llasa_score
        average_category_score = total_llasa_score/SAMPLE_PER_CATEGORY
        average_score_all_categories+=average_category_score
        assessments_f.write(f"AVERAGE SCORE FOR CATEGORY {category}: {average_category_score}\n")

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
    average_score_all_categories = average_score_all_categories / len(categories)
    assessments_f.write(f"AVERAGE SCORE: {average_score_all_categories}")