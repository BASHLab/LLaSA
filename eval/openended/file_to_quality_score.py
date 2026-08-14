import random
import os
import json
import csv

import numpy as np
from tqdm import tqdm

from openai import OpenAI


client = OpenAI()


def quality_scoring(q, standard_answer, predicted_answer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at assessing the quality of predicted answers "
                "based on questions related to IMU data and given information. "
                "Please use the 'Activity Label', 'Sensor Location', and 'Standard Answer' "
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
                f"{standard_answer}\n"
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


random.seed(42)
np.random.seed(42)

model = "claude"

assessments_f = open(f'/home/simran/SLU/results/openended/assessments_{model}_10.txt','w')
total_gpt_score = 0

with open("/home/simran/SLU/eval/openended/standard.txt","r") as f:
    standard_answers = f.readlines()

with open(f"/home/simran/SLU/eval/openended/{model}.txt","r") as f:
    predicted_answers = f.readlines()

with open("/home/simran/SLU/eval/human/prompts.txt","r") as f:
    questions = f.readlines()

total_score = 0

for q, standard_answer, predicted_answer in zip(questions, standard_answers, predicted_answers):
    q = q.split("[[")[0].strip()
    gpt_score_explanation = quality_scoring(
        q, standard_answer, predicted_answer
    )
    print(gpt_score_explanation)
    
    if gpt_score_explanation.split()[2].isnumeric():
        gpt_score = int(gpt_score_explanation.split()[2])
    else:
        print(f"Nonnumerical score")

    assessments_f.write(predicted_answer+"\n")
    assessments_f.write(gpt_score_explanation+"\n-----------------------------------\n")
    
    
    if isinstance(gpt_score, int):
        total_score += gpt_score
average_score = total_score/10
assessments_f.write(f"AVERAGE SCORE: {average_score}\n")