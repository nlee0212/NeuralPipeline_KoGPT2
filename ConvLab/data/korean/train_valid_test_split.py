import numpy as np
import json


def split(corpus, ratio=(0.8, 0.1, 0.1), shuffle=True, random_state=1004):
    train_size, valid_size, test_size = ratio
    test_num = int(len(corpus) * test_size)
    valid_num = int(len(corpus) * valid_size)
    train_num = len(corpus) - test_num - valid_num

    np.random.seed(random_state)
    shuffled = np.random.permutation(len(corpus))
    corpus = np.array(corpus)
    corpus = corpus[shuffled]
    train_dataset = corpus[:train_num]
    valid_dataset = corpus[train_num:train_num + valid_num]
    test_dataset = corpus[train_num + valid_num:]

    return train_dataset, valid_dataset, test_dataset


def modify_dialog_acts(filename, corpus):
    topic = filename.split("_")[0]
    for session in corpus:
        for utter in session["utters"]:
            if topic == "weather":
                utter["dialog_acts"] = utter["dialog_acts"][0]
                utter['text'] = utter['text'][0]
            for dialog_act in utter["dialog_acts"]:
                act = dialog_act["act"]
                if '-' in act:
                    # print(dialog_act['act'])
                    dialog_act['act'] = act.replace('-', '_')
                    # print(dialog_act['act'])
                    act = dialog_act['act']
                dialog_act["act"] = topic + "-" + act


file_name = ["contact_corpus_20190820.json", "schedule_corpus_ext_20181001.json", "weather_corpus_20190125.json"]
train_dataset, valid_dataset, test_dataset = [], [], []
for file in file_name:
    with open(file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    modify_dialog_acts(file, json_data)
    train, valid, test = split(json_data)
    train_dataset += [session for session in train]
    valid_dataset += [session for session in valid]
    test_dataset += [session for session in test]

new_set = []
for set in [train_dataset, valid_dataset, test_dataset]:
    set = np.array(set)
    np.random.seed(1004)
    shuffled = np.random.permutation(len(set))
    set = set[shuffled]
    new_set.append(set)

train_dataset, valid_dataset, test_dataset = (set.tolist() for set in new_set)

with open("korean_train.json", "w", encoding='utf-8') as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent="  ")

with open("korean_valid.json", "w", encoding='utf-8') as f:
    json.dump(valid_dataset, f, ensure_ascii=False, indent="  ")

with open("korean_test.json", "w", encoding='utf-8') as f:
    json.dump(test_dataset, f, ensure_ascii=False, indent="  ")
