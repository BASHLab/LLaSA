import json
import os
import numpy as np
from tqdm import tqdm
import random
import re
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# random.seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PER_CLASS_SAMPLES = 100


unique_classes_dict = {}
unique_classes =[]
with open("/hdd/LLM/hapt/activity_labels.txt") as f:
    contents = f.readlines()
    for i in range(len(contents)):
        content = contents[i].strip()
        content_array = content.split(" ")
        idx_ = int(content_array[0])
        label = content_array[1]
        unique_classes_dict[label] = idx_
        unique_classes.append(label)

unique_classes = set(unique_classes)
print(unique_classes)

UNCLEAR_LABEL = "Unclear"
unique_classes_dict[UNCLEAR_LABEL] = i +2

print(unique_classes_dict)



data_file_name = "/hdd/LLM/hapt/data_20Hz_hapt.npy"
label_file_name = "/hdd/LLM/hapt/label_20Hz_hapt.npy"


# model_path = "/hdd/LLM/llava_llasa_first_test_model"
model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-lora"
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
# image_file = "/hdd/LLM/limuBERT_data/train_llasa/images/sensor_image_20000_1.npy"


result_f = open("/hdd/LLM/SLU/results/llasa_hapt.txt","w")
# sample_indices = [5, 9, 12, 1199, 2000]


pred = []
gt = []

data = np.load(data_file_name)
label = np.load(label_file_name)
label_dict = {v: k for k, v in unique_classes_dict.items()}
print(label_dict)
label_count_dict = {}
for val in unique_classes_dict.keys():
    label_count_dict[val] = 0
print(label_count_dict)
# for idx in tqdm(range(len(data))):
#     d_ = data[idx]
#     np.save('../../hapt/images/hapt_image_' + str(idx)+'.npy', d_)
    # break
for idx in tqdm(range(len(data))):
    # sample_index = random.randint(0, len(data))
    sample_index = idx
    image_file = f"/hdd/LLM/hapt/images/hapt_image_{sample_index}.npy"

    if not os.path.exists(image_file):
        continue
    
    sample_label = label_dict[int(label[sample_index])]
    if label_count_dict[sample_label] < PER_CLASS_SAMPLES:
        label_count_dict[sample_label]+=1
    else:
        continue
    result_f.write(image_file+"\n")
    args = type('Args', (), {
        "model_path": model_path,
        # "model_base": None,
        "model_base": "lmsys/vicuna-7b-v1.5",
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
    outputs = eval_model(args)
    result_f.write(f"{outputs}\n")
    
    gt.append(unique_classes_dict[sample_label])
    if answer_format not in outputs:
        result_f.write("INSTRUCTION AVOIDED\n")
        pred.append(unique_classes_dict[UNCLEAR_LABEL])
    else:
        pred_label = outputs.split(answer_format)[-1]
        pred_label = re.sub(r'[^\w\s]','', pred_label)
        pred_label = pred_label.strip()
        if pred_label in unique_classes_dict.keys():
            pred.append(unique_classes_dict[pred_label])
        else:
            result_f.write("CLASS OUT OF BOX\n")
            pred.append(unique_classes_dict[UNCLEAR_LABEL])

result_f.write(classification_report(
    gt, pred, 
    labels=list(range(len(unique_classes_dict))),
    target_names=unique_classes_dict.keys())
)

cm = confusion_matrix(gt, pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=label_count_dict.keys()
)
disp.plot()
plt.savefig("/hdd/LLM/SLU/results/hapt"+"_cm.png")
