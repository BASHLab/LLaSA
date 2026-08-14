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
                "Assume that you are an expert at assessing the quality of predicted answers based on questions related to IMU data and given information. Please use the ‘Standard Answer,’ ‘Activity Label,’ and ‘Sensor Location’ to assess the ‘Predicted Answer’ based on the correctness, completeness, consistency, and helpfulness of the predicted answer. Every question is given accelerometer and gyroscope data. So, do not only look at the general answer; check if the answer provided any insights from the data. Provide a single overall score for each answer on a scale of 0 to 100 using the following format: ‘Quality score:’ and then, give a brief summary of the assessment in 3-4 sentences, followed by ‘Assessment:’. Be specific in evaluating the accuracy, detail, and relevance of the predicted answer in relation to the expected standard."
                # "You are an expert at assessing the quality of predicted answers "
                # "based on questions related to IMU data and given information. "
                # "Please use the 'Standard Answer' "
                # "to assess the 'Predicted Answer' "
                # "based on the correctness, completeness, consistency, and helpfulness "
                # "of the predicted answer. "
                # "Give a single overall score in the following "
                # "format on a scale of 0 to 100:\n"
                # "Quality score: \n"
                # "Then give a short summary of the assessment in a couple of sentences "
                # "followed by 'Assessment:'. "
                # "Finally, provide a verdict whether the answer is good or bad followed by "
                # "'Verdict:'."
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

model = "7b_rescored"

assessments_f = open(f'/home/simran/SLU/results/openended/assessments_{model}.txt','w')
total_gpt_score = 0


standard_answers = []
predicted_answers = []
questions = []
with open("/home/simran/SLU/results/openended/assessments_human_llava-v1.5-13b-llasa2-lr3e-5-mmlr1e-4-e1-r8-a8-proj1-lora.txt","r") as f:
    all_lines = f.readlines()

for line in all_lines:
    line = line.strip()
    # if "npy" in line:
    #     activity_label = line.split(".")[0].split("_")[1]
    if len(line) == 0:
        continue
    if line[-1] == "?" or "Evaluate the gyroscope data for measuring" in line:
        questions.append(line)
    elif "LLaSA:" in line:
        predicted_answers.append(line.split("LLaSA:")[-1])
    elif "Standard:" in line:
        standard_answers.append(
            (
                # f"Activity Label: {activity_label}\n"
                # f"Sensor Location: hand\n"
                f'Standard Answer: {line.split("Standard:")[-1]}'
            )
        )

# print(len(questions), len(predicted_answers), len(standard_answers))
# print(questions[50])
# print(predicted_answers[50])
# print(standard_answers[50])

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
average_score = total_score/100
assessments_f.write(f"AVERAGE SCORE: {average_score}\n")