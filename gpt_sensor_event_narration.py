import base64
import json
import os

import cv2
from openai import OpenAI

from tqdm import tqdm
import pandas as pd


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
    "Narrate the video in 15 sentences with details that are "
    "context-aware based on the sensor data."
)


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


def generate_llava_finetune_narration_data(
    samples=2, frame_subsample=10, sensor_subsample=8
):
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
    
    df = available_train_df.sample(n=samples)
    for _, row in tqdm(df.iterrows()):
        participant_id = row["participant_id"]
        video_id = row["video_id"]
        narration_id = row["narration_id"]
        frame_dir = os.path.join(data_dir,participant_id,"rgb_frames",video_id)
        frames = sorted(os.listdir(frame_dir))
        accl = pd.read_csv(
            f"/hdd/LLM/EK-Data/{participant_id}/meta_data/{video_id}-accl.csv"
        ).round(3)
        gyro = pd.read_csv(
            f"/hdd/LLM/EK-Data/{participant_id}/meta_data/{video_id}-gyro.csv"
        ).round(3)
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
        accl_segment_list = accl_segment.drop(
            ["Milliseconds"], axis=1
        ).values.tolist()[::sensor_subsample]
        accl_str = str(accl_segment_list)
        gyro_segment_list = gyro_segment.drop(
            ["Milliseconds"], axis=1
        ).values.tolist()[::sensor_subsample]
        gyro_str = str(gyro_segment_list)
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
            f"Summary: {summary}. "
            f"Gyroscope: {gyro_str} "
            f"Accelerometer: {accl_str}"
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
            "model": "gpt-4-turbo",
            "messages": prompt_messages,
            "max_tokens": 300,
        }

        result = client.chat.completions.create(**params)
        narration = result.choices[0].message.content
        print(narration)

        train_json_list.append(
            {
                "id": f"{narration_id}",
                "video": video_name,
                "conversations": [
                    {"from": "human", "value": f"<video>\n{prompt}"},
                    {"from": "gpt", "value": narration},
                ],
            }
        )
    with open("sensor_event_narration_llava_train.json", "w") as f:
        json.dump(train_json_list, f)


if __name__ == '__main__':
    generate_llava_finetune_narration_data()
