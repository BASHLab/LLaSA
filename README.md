[![Hippocratic License HL3-CL-ECO-EXTR-FFD-LAW-MEDIA-MIL-MY-SOC-SV-TAL-USTA](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-EXTR-FFD-LAW-MEDIA-MIL-MY-SOC-SV-TAL-USTA&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-eco-extr-ffd-law-media-mil-my-soc-sv-tal-usta.html)

# LLaSA: A Multimodal Large Language Model for Human Activity Analysis Using Sensor Data

[LLaSA Demo](llasa_demo.png "A comparison between GPT-3.5 Turbo and LLaSA when responding to a query about potential obstacles encountered while vacuuming. GPT-3.5 Turbo provides a generalized instruction on analyzing IMU data, while LLaSA directly interprets the data, identifying specific sensor readings like high peaks in z-axis acceleration and rapid gyroscope changes to detect obstacles. This showcases LLaSA’s ability to offer precise, data-driven, and contextually relevant answers.")

This repository hosts the code and datasets for **LLaSA (Large Language and Sensor Assistant)**, a multimodal large language model that integrates inertial measurement units (IMUs) with natural language understanding. Built on **LIMU-BERT** and **Llama**, LLaSA is designed to interpret and respond to complex queries about human activities and motion by combining sensor data with contextual reasoning.

## Key Features

### 1. Datasets
- **SensorCaps**: A dataset of 35,960 IMU-derived activity narrations enriched with handcrafted features.
- **OpenSQA**: An instruction-following dataset containing 179,727 question-answer pairs, tailored for sensor- and activity-aware contexts.

### 2. Model Architecture
- LLaSA integrates IMU data with natural language processing capabilities, leveraging multimodal inputs for nuanced activity analysis.
- Includes advanced hyperparameter tuning to optimize performance for contextual, sensor-based question-answering tasks.

### 3. Evaluation
- Comprehensive evaluations, including human-led assessments, show that LLaSA outperforms GPT-3.5-Turbo and Vicuna-1.5-13b-16K in sensor-aware and context-sensitive question answering.
- We employed a hyperparameter tuning method with GPT-assisted evaluation of question-answer pairs.

## Applications
LLaSA is designed to support impactful research and practical applications in:
- **Personal Health**: Monitoring activity patterns, providing actionable insights, and assisting in wellness routines.
- **Human-Computer Interaction**: Context-aware assistance and enhanced user experience through activity interpretation.

## Repository Contents
- **Code**: Scripts for training, fine-tuning, and evaluating the LLaSA model.
- **Datasets**: SensorCaps and OpenSQA [Google Drive](https://drive.google.com/drive/folders/1128HH_idfgmZnDeqQV2rjFhm4-yW7HK7?usp=share_link) (Email address: llasa.data@gmail.com)
- **Documentation**: Instructions for replicating experiments and integrating LLaSA into your projects.


