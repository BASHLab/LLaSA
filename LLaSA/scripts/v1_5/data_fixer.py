with open("../llasa_training_v2_raw.jsonl","r") as f:
    s = f.read()


s = s.replace("/hdd/LLM/limuBERT_data/train_llasa/images/","/home/simran/limuBERT_data/train_llasa/images/")

with open("../llasa_training_v2_raw_fixed.jsonl","w") as f:
    f.write(s)