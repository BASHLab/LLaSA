import base64
import json
import os

import cv2
from openai import OpenAI

from tqdm import tqdm
import pandas as pd
import numpy as np


client = OpenAI()

AVAILABLE_PARTICIPANTS = ['P01', 'P02', 'P03']
INSTRUCTION = (
    "Please consider yourself to be an expert "
    "on gyroscope and accelerometer sensor information "
    "given as a metadata in a vision dataset. "
    "You are given the frames of an egocentric video "
    "in a home environment setting. "
    "The user also provides a brief summary of the event "
    "followed by 'Summary:'. "
    "They also give you gyroscopic and accelerometer sensor data followed by "
    "'Gyroscope:' and 'Accelerometer:' respectively. "
    "They are written in a Python list of lists format and contain "
    "x, y, and z axis data respectively. "
    "Additional information computed for each axis is provided afterwards, "
    "with the windows for moving averages being 12. "
    "Generate 5 question-answer pairs that require thorough analysis "
    "based on the context of the video and the sensor data."
    "Write the QA pairs in the following format: "
    "1. Q: \n A: \n 2. Q: \n A: \n"
)

DATA_EVAL_INSTRUCTION = (
    "You assess the quality of the question-answer pairs for a dataset "
    "generated with context from IMU sensor and video data. "
    "Give it an overall mean score out of 100. "
    "Consider qualities such as details, reliability, helpfulness, "
    "lack of hallucination, relevance, completeness, etc. when scoring. "
    "You are given the frames of an egocentric video "
    "in a home environment setting. "
    "The user also provides a brief summary of the event "
    "followed by 'Summary:'. "
    "They also give you gyroscopic and accelerometer sensor data followed by "
    "'Gyroscope:' and 'Accelerometer:' respectively. "
    "The 5 QA pairs that you would score together are generated via LLM and "
    "are included after 'QAs:'. "
    "Respond in the following format: "
    "'Overall Score: (integer)\nExplanation:'"
)


def moving_average(a, n=12):
    ma = np.zeros((a.shape[0]-n+1, a.shape[1]), dtype=float)
    for i in range(a.shape[1]):
        ma[:,i] = np.convolve(a[:,i], np.ones(n), 'valid') / n
    return ma


def to_millisecond(str_):
    s = str_.split(":")
    sum_ = 0
    for i, ss in enumerate(s[::-1]):
        sum_ += float(ss)*(60**i)*1000
    return sum_


def generate_video(images, image_folder, video_name):
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        frame = cv2.resize(frame, (width, height))
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()

def round_str(a):
    return str(np.round(a, 6))


def generate_llava_finetune_narration_data(
    samples=10, frame_subsample=10, sensor_subsample=8
):
    score_f = open("epickitchen_qa_data_score.txt", "w")
    total_score = 0
    
    train_json_list = []

    data_dir = "/hdd/LLM/EK-Data"
    train_df = pd.read_csv("/hdd/LLM/EK-annotations/EPIC_100_train.csv")
    
    available_data_dict = {}
    for participant_id in AVAILABLE_PARTICIPANTS:
        video_names = sorted(os.listdir(os.path.join(data_dir, participant_id, "videos")))
        video_ids = []
        for video_name in video_names:
            video_ids.append(video_name.split(".")[0])
        available_data_dict[participant_id] = video_ids
    
    available_train_df = pd.DataFrame()
    for participant_id, video_ids in available_data_dict.items():
        participant_df = train_df[train_df["participant_id"]==participant_id]
        for video_id in video_ids:
            video_df = participant_df[participant_df["video_id"]==video_id]
            available_train_df = pd.concat([available_train_df, video_df], ignore_index=True, axis=0)
    
    df = available_train_df.sample(n=samples, random_state=42)
    for _, row in tqdm(df.iterrows()):
        participant_id = row["participant_id"]
        video_id = row["video_id"]
        narration_id = row["narration_id"]
        frame_dir = os.path.join(data_dir,participant_id,"rgb_frames",video_id)
        frames = sorted(os.listdir(frame_dir))
        accl = pd.read_csv(
            f"/hdd/LLM/EK-Data/{participant_id}/meta_data/{video_id}-accl.csv"
        )
        gyro = pd.read_csv(
            f"/hdd/LLM/EK-Data/{participant_id}/meta_data/{video_id}-gyro.csv"
        )
        start_timestamp = row["start_timestamp"]
        stop_timestamp = row["stop_timestamp"]
        start_time_ms = to_millisecond(str(start_timestamp))
        stop_time_ms = to_millisecond(str(stop_timestamp))
        accl_segment = accl[
            accl["Milliseconds"] > start_time_ms
        ][accl["Milliseconds"] < stop_time_ms]
        gyro_segment = gyro[
            gyro["Milliseconds"] > start_time_ms
        ][gyro["Milliseconds"] < stop_time_ms]
        
        accl_np = np.array(accl_segment)
        gyro_np = np.array(gyro_segment)
        
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
        
        accl_segment_list = accl_segment.drop(
            ["Milliseconds"], axis=1
        ).values.tolist()[::sensor_subsample]
        accl_str = round_str(accl_segment_list)
        gyro_segment_list = gyro_segment.drop(
            ["Milliseconds"], axis=1
        ).values.tolist()[::sensor_subsample]
        gyro_str = round_str(gyro_segment_list)
        print(f"Sensor len: {len(accl_segment_list)} {len(gyro_segment_list)}")

        base64_frames = []
        start_frame = row["start_frame"]-1
        stop_frame = row["stop_frame"]
        frames = frames[start_frame:stop_frame][::frame_subsample]
        print(f"Frame len: {len(frames)}")
        for frame in tqdm(frames, leave=False):
            buffer = cv2.imread(os.path.join(frame_dir, frame))
            _, buffer = cv2.imencode(".jpg", buffer)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        video_name = f"/hdd/LLM/finetune_data/gpt4_videos/{narration_id}.MP4"
        generate_video(frames, frame_dir, video_name)

        summary = row["narration"]

        prompt = (
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
            f"Accelerometer variance: {round_str(accl_var)}"
        )
        prompt_messages = [
            {
                "role": "system",
                "content": INSTRUCTION
            },
            {
                "role": "user",
                "content": [
                    prompt,
                    *map(lambda x: {"image": x}, base64_frames),
                ],
            },
        ]
        params = {
            "model": "gpt-4o-mini",
            "messages": prompt_messages,
            "max_tokens": 2000,
        }

        result = client.chat.completions.create(**params)
        qas = result.choices[0].message.content
        print(qas)

        train_json_list.append(
            {
                "id": f"{narration_id}",
                "video": video_name,
                "conversations": [
                    {"from": "human", "value": f"<video>\n{prompt}"},
                    {"from": "gpt", "value": qas},
                ],
            }
        )
        
        data_eval_prompt = (
            f"Summary: {summary}\n"
            f"Gyroscope: {gyro_str}\n"
            f"Accelerometer: {accl_str}\n"
            f"QAs: {qas}"
        )
        data_eval_messages = [
            {
                "role": "system",
                "content": DATA_EVAL_INSTRUCTION
            },
            {
                "role": "user",
                "content": [
                    data_eval_prompt,
                    *map(lambda x: {"image": x}, base64_frames),
                ],
            },
        ]
        
        data_eval_params = {
            "model": "gpt-4o",
            "messages": data_eval_messages,
            "max_tokens": 300,
        }
        
        data_eval_result = client.chat.completions.create(**data_eval_params)
        data_score = data_eval_result.choices[0].message.content
        print(data_score)
        score_int = data_score.lower().split("overall score: ")[-1]
        score_int = score_int.split("\n")[0]
        total_score += int(score_int)
        
        score_f.write(qas+"\n"+"-"*20+"\n")
        score_f.write(data_score+"\n")
        
        
    with open("epickitchen_qa_llasa_train.json", "w") as f:
        json.dump(train_json_list, f)
    
    mean_score = total_score / samples
    score_f.write(f"AVERAGE: {mean_score}")


if __name__ == '__main__':
    generate_llava_finetune_narration_data()
