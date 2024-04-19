"""
It's notable that this pipeline verified the limitations of
current open-source LLMs. They can't understand sensor
data like GPT-4 can. Therefore, we should come back to this
script after finetuning our model.
"""


import os

import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import Conversation, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)


import pandas as pd

disable_torch_init()


INSTRUCTION = (
    "Please consider yourself to be an expert "
    "on gyroscope and accelerometer sensor information "
    "given as a metadata in a vision dataset. "
    "You are given an egocentric video in a home environment setting. "
    "The user also provides a brief summary of the event "
    "followed by 'Summary:'. "
    "They also give you gyroscopic and accelerometer sensor data followed by "
    "'Gyroscope:' and 'Accelerometer:' respectively. "
    "They are written in a Python list of lists format and contain "
    "x, y, and z axis data respectively. "
    "Narrate the video with details that are "
    "context-aware based on the sensor data."
)


model_path = 'LanguageBind/Video-LLaVA-7B'
cache_dir = 'cache_dir'
device = 'cuda'
load_4bit, load_8bit = False, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, _ = load_pretrained_model(
    model_path, None, model_name, load_8bit, load_4bit,
    device=device, cache_dir=cache_dir
)
video_processor = processor['video']


def narrate_sensor_event(video_filename, video_dir, sensor_dir, summary_dir, subsample=25):
    video = os.path.join(video_dir, video_filename)
    imu_filename = video_filename.replace(".MP4", ".csv")
    summary_filename = video_filename.replace(".MP4", ".txt")
    imu_file = os.path.join(sensor_dir, imu_filename)
    summary_file = os.path.join(summary_dir, summary_filename)
    imu_df = pd.read_csv(imu_file).round(3)
    accl_str = str(imu_df[['AcclX', 'AcclY', 'AcclZ']].values.tolist()[::subsample])
    gyro_str = str(imu_df[['GyroX', 'GyroY', 'GyroZ']].values.tolist()[::subsample])
    print(len(imu_df.values.tolist()[::subsample]))
    if len(imu_df.values.tolist()[::subsample])>15:
        return "Too long"
    with open(summary_file, "r") as f:
        summary = f.read()
    inp = (
        f"{INSTRUCTION}\n"
        f"Summary: {summary}\n"
        f"Gyroscope: {gyro_str}\n"
        f"Accelerometer: {accl_str}\n"
    )
    conv = Conversation(
        system=INSTRUCTION,
        roles=("USER", "ASSISTANT"),
        version="v1",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    ).copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if isinstance(video_tensor, list):
        tensor = [
            video.to(
                model.device, dtype=torch.float16
            )
            for video in video_tensor
        ]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join(
        [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames
    ) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, tokenizer, input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs


if __name__ == '__main__':
    VIDEO_DIR = "/hdd/LLM/finetune_data/videos"
    SENSOR_DIR = "/hdd/LLM/finetune_data/imus"
    SUMMARY_DIR = "/hdd/LLM/finetune_data/event_description"
    video_filenames = os.listdir(VIDEO_DIR)
    with open("results.txt", "w") as f:
        for filename in video_filenames[:3]:
            narration = narrate_sensor_event(
                video_filename=filename,
                video_dir=VIDEO_DIR,
                sensor_dir=SENSOR_DIR,
                summary_dir=SUMMARY_DIR
            )
            f.write(filename+"\n"+narration+"\n")
