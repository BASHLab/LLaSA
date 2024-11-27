from pathlib import Path
import pandas as pd
import os
import re
import numpy as np
import json
from tqdm import tqdm
import random

import json

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch

seed = 42

random.seed(42)
np.random.seed(42)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


PER_CLASS_SAMPLES = 10

random.seed(42)

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
        # indices_20hz = np.round(np.linspace(0, len(data_chunk) - 1, 120)).astype(int)
        # data_chunk_subsampled = data_chunk[indices_20hz]
        npy_name = npy_name_base + f"_{npy_idx}.npy"
        # np.save(os.path.join(data_dir, npy_name), data_chunk_subsampled)
        
        # accl = data_chunk_subsampled[:,0:3]
        # gyro = data_chunk_subsampled[:,0:6]
        # accl, gyro = accl.tolist(), gyro.tolist()
        # accl_str = sensor_subsampled_string(accl)
        # gyro_str = sensor_subsampled_string(gyro)
        
        npy_file_to_string[npy_name] = label_name
    
    # subset = dict(random.choices(list(npy_file_to_string.items()),k=2))
    subset = npy_file_to_string
    return subset


npy_file_to_string_all = {}
# num_user_class=0
for path in Path('/home/simran/SLU/eval/human/llasa_data').rglob('*Accelerometer.csv'):
    npy_file_to_string = subsample_data(path)
    npy_file_to_string_all.update(npy_file_to_string)
    # num_user_class+=1
# print(num_user_class)

# unique_classes = set()
unique_classes = set(label_name_dict.values())
print(unique_classes)


unique_classes_dict = {}

for idx, unique_class in enumerate(unique_classes):
    unique_classes_dict[unique_class] = idx

UNCLEAR_LABEL = "Unclear"
unique_classes_dict[UNCLEAR_LABEL] = idx+1

print(unique_classes_dict)

answer_format = "The identified class is: "
prompt = (
    "The given IMU sensor data can be associated with one of the following classes: "
    f"{unique_classes}. "
    "Write only the name of the identified class in the format "
    f"'{answer_format}'"
)
print(prompt)
a=0/0

dataset = "human"
result_f = open("/home/simran/SLU/results/human_classify.txt","w")
model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa_v1"

pred = []
gt = []

label_dict = unique_classes_dict

label_count_dict = {}
for val in label_dict.keys():
    label_count_dict[val] = 0
data = list(npy_file_to_string_all.items())
for idx in tqdm(range(len(data))):
    sample_index = random.randint(0, len(data))
    
    # print(data[sample_index])
    image_file, sample_label = data[sample_index]
    image_file = os.path.join("/home/simran/SLU/eval/human/bash_qa_base",image_file)
    
    if label_count_dict[sample_label] < PER_CLASS_SAMPLES:
        label_count_dict[sample_label]+=1
    else:
        continue
    result_f.write(image_file+"\n")
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 30
    })()
    result_f.write(f"True: {sample_label}\n")
    
    outputs = eval_model(args)
    print(outputs)
    print("===============================")
    result_f.write(f"{outputs}\n")
    
    gt.append(unique_classes_dict[sample_label])
    if answer_format not in outputs:
        result_f.write("INSTRUCTION AVOIDED\n")
        pred.append(unique_classes_dict[UNCLEAR_LABEL])
        if UNCLEAR_LABEL not in label_count_dict.keys():
            label_count_dict[UNCLEAR_LABEL] = 0
        else:
            label_count_dict[UNCLEAR_LABEL] += 1
    else:
        pred_label = outputs.split(answer_format)[-1]
        pred_label = re.sub(r'[^\w\s]','', pred_label)
        pred_label = pred_label.strip().lower()
        if pred_label in unique_classes_dict.keys():
            pred.append(unique_classes_dict[pred_label])
        else:
            result_f.write("CLASS OUT OF BOX\n")
            pred.append(unique_classes_dict[UNCLEAR_LABEL])
            if UNCLEAR_LABEL not in label_count_dict.keys():
                label_count_dict[UNCLEAR_LABEL] = 0
            else:
                label_count_dict[UNCLEAR_LABEL] += 1
    
    print(f"index: {sample_index} pred: {pred[-1]}, gt: {gt[-1]}")

result_f.write(classification_report(
    gt, pred, 
    labels=list(range(len(unique_classes_dict))),
    target_names=unique_classes_dict.keys())
)

# cm = confusion_matrix(gt, pred)
ConfusionMatrixDisplay.from_predictions(
    gt, pred, 
    labels=list(range(len(unique_classes_dict))),
    display_labels=unique_classes_dict.keys(),
    xticks_rotation='vertical'
)
plt.tight_layout()
plt.savefig("/home/simran/SLU/results/"+dataset+"_cm.png", pad_inches=5)
