import os
import csv
import re

from openai import OpenAI

SAMPLE_PER_CATEGORY = 50

# old_client = OpenAI()

models_and_clients = {
    "deepseek-chat": OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"),
    "vicuna-13b-v1.5-16k": OpenAI(
        base_url = "http://localhost:8000/v1/",
        api_key = "EMPTY"
    ),
    
}


csv_file =  open('/home/simran/SLU/results/openended/interannotator.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["sample","rater","score"])

categories = ["narration", "reliability", "math"]
benchmark_dir = "/hdd/LLM/PAMAP2/qa_benchmark/"
image_dir = "/" 

# model_path = "/hdd/shouborno/llava-experiment/LLaVA/checkpoints/llava-v1.5-13b-lora"
# model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa_v2_small"
# model_path = "/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa2-lr1e-4-mmlr1e-4-e1-r8-a8-lora"

checkpoint_dir = "/home/simran/SLU/LLaSA/checkpoints3/"
model_names = [
    # "llava-v1.5-13b-llasa2-lr3e-5-mmlr1e-4-e1-r8-a8-proj1-lora"
    dirname 
    for dirname in os.listdir(checkpoint_dir) 
    if dirname.startswith('llava-v1.5-13b-llasa2-lr')
]

print(model_names)


def quality_scoring(q, standard_answer, activity_label, sensor_location, predicted_answer, client, model):
    messages = [
        {
            "role": "system",
            "content": (
                "Assume that you are an expert at assessing the quality of predicted answers based on questions related to IMU data and given information. Please use the ‘Standard Answer,’ ‘Activity Label,’ and ‘Sensor Location’ to assess the ‘Predicted Answer’ based on the correctness, completeness, consistency, and helpfulness of the predicted answer. Every question is given accelerometer and gyroscope data. So, do not only look at the general answer; check if the answer provided any insights from the data. Provide a single overall score for each answer on a scale of 0 to 100 using the following format: ‘Quality score:’ and then, give a brief summary of the assessment in 3-4 sentences, followed by ‘Assessment:’. Be specific in evaluating the accuracy, detail, and relevance of the predicted answer in relation to the expected standard."
                # "You are an expert at assessing the quality of predicted answers "
                # "based on questions related to IMU data and given information. "
                # "Please use the 'Standard Answer', 'Activity Label', and 'Sensor Location' "
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
                f"Standard Answer: {standard_answer}\n"
                f"Activity Label: {activity_label}\n"
                f"Sensor Location: {sensor_location}\n"
                f"Predicted Answer: {predicted_answer}"
            ),
        },
    ]
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": 300,
    }

    result = client.chat.completions.create(**params)
    outputs = result.choices[0].message.content
    return outputs

# result_summary_f = open('/home/simran/SLU/results/openended/summary.txt','w')


sample_count = 0
for model_name in model_names:
    if sample_count > SAMPLE_PER_CATEGORY:
        break
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    
    
    average_score_all_categories = 0
    category_count = 0
    total_llasa_score = 0
    total_numerical_scores = 0
    try:
        with open(f'/home/simran/SLU/results/openended/assessments_5_{model_name}.txt','r') as old_f:
            old_lines = old_f.readlines()
    except:
        print(f"{model_name} missing")
        continue
    for old_line in old_lines:
        if old_line.startswith("/"):
            activity_label = old_line.split("_")[-2]
            sensor_location = old_line.split("_")[-3].split("/")[-1]
        elif old_line.startswith("LLaSA:"):
            q = q_candidate
            llasa_answer = old_line.split("LLaSA: ")[-1]
        elif old_line.startswith("Standard:"):
            standard_answer = old_line.split("Standard: ")[-1]
            # if sample_count%SAMPLE_PER_CATEGORY==0:
            #     sample_count = 0
            #     total_llasa_score = [0]*len(categories)
            #     total_numerical_scores = 0
            # else:
            sample_count+=1
            llasa_score = [0]
            for rater_model, client in models_and_clients.items():
                
                llasa_score_explanation = quality_scoring(
                    q, standard_answer, activity_label, sensor_location, llasa_answer, client, rater_model
                )
                print(llasa_score_explanation)
                
                # assessments_f.write(
                #     filepath+"\n"+q+"\nLLaSA: "+str(llasa_answer.encode("utf8"))
                #     +"\nStandard: "+standard_answer+"\n"
                # )
                # assessments_f.write(rater_model+": "+llasa_score_explanation+"\n-----------------------------------\n")
                numbers_in_explanation = re.findall(r'\d+', llasa_score_explanation.split("\n")[0])
                if (
                    "quality score" in llasa_score_explanation.lower() 
                    and 
                    len(numbers_in_explanation)==1
                ):
                    llasa_score_by_rater = numbers_in_explanation[0]
                else:
                    print(f"Nonnumerical score")
                llasa_score.append(llasa_score_by_rater)
                writer.writerow([sample_count,rater_model,llasa_score_by_rater])
            # total_llasa_score += llasa_score
        elif old_line.startswith("Quality score:"):
            llasa_score[0] += int(old_line.split("Quality score: ")[-1].strip())
            writer.writerow([sample_count,"gpt-4o",llasa_score[0]])
        # average_category_score = total_llasa_score/SAMPLE_PER_CATEGORY
        # average_score_all_categories+=average_category_score

        category = categories[category_count]
        q_candidate = old_line
        
        # average_category_score = total_llasa_score/SAMPLE_PER_CATEGORY
        # average_score_all_categories+=average_category_score
        
    
    
    # model_path = os.path.join(checkpoint_dir, model_name)
    # for category in categories:
        
        
    #     # csvfile = open(f"/home/simran/SLU/results/openended/{category}_results.csv", 'w')
    #     # csvwriter = csv.writer(csvfile)
    #     # csvwriter.writerow(fields)
    #     print(os.path.join(benchmark_dir,f'{category}_test.jsonl'))
    #     print(os.path.exists(os.path.join(benchmark_dir,f'{category}_test.jsonl')))
    #     category_test_data = json.load(
    #         open(os.path.join(benchmark_dir,f'{category}_test.jsonl'),'r')
    #     )
    #     print(len(category_test_data))
    #     category_test_data = random.sample(category_test_data, SAMPLE_PER_CATEGORY)
    #     for sample in tqdm(category_test_data):
    #         sensor_location = sample["id"].split("_")[0]
    #         activity_label = sample["id"].split("_")[1]
    #         filepath = sample["image"]
    #         q = sample["conversations"][0]["value"].replace("<image>\n", "")
    #         standard_answer = sample["conversations"][1]["value"]
            
    #         data = np.load(os.path.join(image_dir,filepath))
            
    #         accl, gyro = data[:, 0:3], data[:, 3:6]
    #         accl, gyro = accl.tolist(), gyro.tolist()
    #         accl_str = sensor_subsampled_string(accl)
    #         gyro_str = sensor_subsampled_string(gyro)

    #         # gpt_answer = get_gpt_answer(accl_str, gyro_str, q)
    #         # gpt_score_explanation = quality_scoring(
    #         #     q, standard_answer, activity_label, sensor_location, gpt_answer
    #         # )
    #         # print(gpt_score_explanation)
            
    #         # if gpt_score_explanation.split()[2].isnumeric():
    #         #     gpt_score = int(gpt_score_explanation.split()[2])
    #         # else:
    #         #     print(f"Nonnumerical score")
            
    #         data[:,0:3] = data[:,0:3]/9.8
    #         temp_filepath = "/home/simran/SLU/temp.npy"
    #         np.save(temp_filepath, data)
            
    #         args = type('Args', (), {
    #             "model_path": model_path,
    #             "model_base": "lmsys/vicuna-7b-v1.5",
    #             # "model_base": None,
    #             "model_name": get_model_name_from_path(model_path),
    #             "query": q,
    #             # "query": f"Gyroscope: {gyro_str} Accelerometer: {accl_str}\n{q}",
    #             "conv_mode": None,
    #             "image_file": temp_filepath,
    #             "sep": ",",
    #             "temperature": 0,
    #             "top_p": None,
    #             "num_beams": 1,
    #             "max_new_tokens": 500
    #         })()
    #         llasa_answer = eval_model(args)
            
            # for rater_model, client in models_and_clients.items():
            #     llasa_score_explanation = quality_scoring(
            #         q, standard_answer, activity_label, sensor_location, llasa_answer, client, rater_model
            #     )
            #     print(llasa_score_explanation)
                
            #     assessments_f.write(
            #         filepath+"\n"+q+"\nLLaSA: "+str(llasa_answer.encode("utf8"))
            #         +"\nStandard: "+standard_answer+"\n"
            #     )
            #     assessments_f.write(rater_model+": "+llasa_score_explanation+"\n-----------------------------------\n")
            #     if llasa_score_explanation.split()[2].isnumeric():
            #         llasa_score = int(llasa_score_explanation.split()[2])
            #     else:
            #         print(f"Nonnumerical score")
            
            
            # if isinstance(llasa_score, int):
            #     total_llasa_score += llasa_score
        
        # assessments_f.write(f"AVERAGE SCORE FOR CATEGORY {category}: {average_category_score}\n")

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
    # average_score_all_categories = average_score_all_categories / len(categories)
    # assessments_f.write(f"AVERAGE SCORE: {average_score_all_categories}")