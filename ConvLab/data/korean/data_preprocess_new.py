import json
import copy
import os
from tqdm import tqdm


# print(DOMAIN)
# print(SLOTS)

def preprocess(trainval):
    train_path = '{}.json'.format(trainval)
    with open(train_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

    iter = 0
    for session in dataset:
        iter += 1
        if iter % 100 == 0:
            print("preprocess iteration [{}/{}]".format(iter, len(dataset)))

        for utter in session['utters']:
            text = utter['text']
            dacts = utter['dialog_acts']

            for dact in dacts:
                # print(dact['act'])
                dom, intent = dact['act'].split('-')
                slot, val = dact['slot'], dact['value']
                if slot in requestables and val != None:
                    dom_value = '[{}_{}]'.format(dom, slot)
                    dact['value'] = dom_value
                    utter['text'] = text.replace(val, dom_value)
                    text = utter['text']

    outfile_name = '{}_delexicalized.json'.format(trainval)
    print('Writing start')
    print(outfile_name)
    with open(outfile_name, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print('Done!')


if __name__ == "__main__":

    with open('domain-requestables.json', 'r', encoding='utf-8') as f:
        domain_requestables = json.loads(f.read())

    requestables = []
    with open('requestables.txt', 'r', encoding='utf-8') as f:
        slots = f.readline()
        slots = slots.split(' ')
        for slot in slots:
            if len(slot) == 0:
                continue
            requestables.append(slot)
    # print(requestables)

    DOMAIN = domain_requestables.keys()
    SLOTS = domain_requestables.values()

    for type in ['test', 'train', 'valid']:
        preprocess("korean_{}".format(type))
