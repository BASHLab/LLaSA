import json
import os
import numpy as np
from tqdm import tqdm
import random
import re

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


with open("/home/simran/limuBERT_data/dataset_activity_label.json","r") as f:
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


root_data_dir = "/home/simran/limuBERT_data/extracted_data"
datasets = sorted(os.listdir(root_data_dir))

data_file_name = "data_20_120.npy"
label_file_name = "label_20_120.npy"

last_axis_dict = {
    "hhar": 2, "motion":0, "shoaib":0, "uci":0
}


model_path = "/home/simran/LLaVA/checkpoints7/llava-v1.5-13b-llasa2-limu4"
# model_path = "/home/simran/LLaVA/checkpoints7/llava-v1.5-13b-llasa2-limu4-lr3e-5-mmlr1e-4-e1-r8-a8-proj1-lora"
# model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa_v1"
# model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa2-lr3e-5-mmlr1e-4-e1-r128-a128-lora"
# model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-llasa_v1"
# model_path = "/home/simran/llava_llasa_first_test_model"
# model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-lora"
# model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-pretrain"

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
# image_file = "/home/simran/limuBERT_data/train_llasa/images/sensor_image_20000_1.npy"


result_f = open("/home/simran/SLU/results/llasa_limubertdatasets_deterministic_13b.txt","w")
# sample_indices = [5, 9, 12, 1199, 2000]

for dataset in datasets:
    pred = []
    gt = []
    
    result_f.write(dataset+"\n")
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
        image_file = f"/home/simran/limuBERT_data/train_llasa/images/{dataset}/sensor_image_{sample_index}.npy"
        if not os.path.exists(image_file):
            continue
        
        sample_label = label_dict[str(int(label[sample_index, 0, last_axis]))]
        if label_count_dict[sample_label] < PER_CLASS_SAMPLES:
            label_count_dict[sample_label]+=1
        else:
            continue
        result_f.write(image_file+"\n")
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            # "model_base": "lmsys/vicuna-13b-v1.5-16K",
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 150
        })()
        result_f.write(f"True: {sample_label}\n")
        
        # import pdb;pdb.set_trace()
        
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
    plt.savefig("/home/simran/SLU/results/13b_"+dataset+"_cm.png", pad_inches=5)
