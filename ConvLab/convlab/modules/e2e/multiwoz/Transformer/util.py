# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import tarfile
import tempfile
import random
import torch
import copy
import re

from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd

from kogpt2_transformers import get_kogpt2_tokenizer

logger = logging.getLogger(__file__)


def get_woz_dataset(tokenizer, dataset_path, dataset_cache=None):
    dataset_path = dataset_path
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__ + '_korean_normalized'

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)

    else:
        logger.info("Download dataset from %s", dataset_path)

        train_path = os.path.join(dataset_path, 'korean_train.json')
        # train_path = os.path.join(dataset_path, 'total_v4.json')
        valid_path = os.path.join(dataset_path, 'korean_valid.json')
        # valid_path = os.path.join(dataset_path, 'val_v4.json')
        with open(train_path, "r", encoding="utf-8") as f:
            train_dataset = json.loads(f.read())
        with open(valid_path, "r", encoding="utf-8") as f:
            valid_dataset = json.loads(f.read())

        def random_candidates(data):
            # randomly choose system's utterance from a randomly chosen dialog
            session = random.choice(data)
            dia = [t['text'].strip() for t in session["utters"]]
            if session["utters"][0]["speaker"] == "System":
                # ignore opening statements from the system
                dia = dia[1:]
            sys_len = len(dia)
            id_ = random.choice(range(sys_len // 2))
            return dia[2 * id_ - 1]

        # print(random_candidates(train_dataset))

        def convert_act(dialog_act):
            # dialog_act: dialog_act list with only system's dialog_acts
            #             each element is in a dictionary form with keys ["act", "slot", "value"]
            bs = []
            sn = set()
            for d in dialog_act:
                tmp = []
                dialog_dict = {}
                for dact in d:
                    if dact['act'] not in dialog_dict.keys():
                        dialog_dict[dact['act']] = []
                    slotvalue = [dact['slot'], dact['value']]
                    dialog_dict[dact['act']].append(slotvalue)
                for k in list(dialog_dict.keys()):
                    tmp.append('<' + k.lower() + '>')
                    sn.add('<' + k.lower() + '>')
                    for slot, value in dialog_dict[k]:
                        if slot == None:
                            slot = "none"
                        if value == None:
                            value = "none"
                        tmp.append('<' + slot.lower() + '>')
                        tmp.append(value.lower())
                        sn.add('<' + slot.lower() + '>')

                bs.append(tmp)

            return bs, sn

        """        
        def convert_meta(dialog_meta, cur_dom):

            cs = []
            for i, d in enumerate(dialog_meta):
                dom = cur_dom[i]

                tmp = []
                if dom == 'none':
                    tmp.append('')
                else:
                    constraint = d[dom]
                    tmp.append('<' + dom.lower() + '>')
                    for b in constraint['book']:
                        if b != 'booked':
                            tmp.append('<' + b.lower() + '>')
                            tmp.append(constraint['book'][b])
                    for s in constraint['semi']:
                        v = constraint['semi'][s]
                        tmp.append('<' + s.lower() + '>')
                        if v in ["dont care", "don't care", "do n't care", "dontcare"]:
                            tmp.append('<dc>')
                        elif v == 'not mentioned':
                            tmp.append('<nm>')
                        else:
                            tmp.append(v)
                cs.append(' '.join(tmp))
            return cs
        """

        def parse_woz_data(data, valid=False):

            dataset = []
            doms = ['contact', 'weather', 'schedule']
            sns = set()
            for dialog in tqdm(data):

                dialog_info = [t['text'].strip() for t in dialog['utters']]
                dialog_act = [t['dialog_acts'] for t in dialog['utters']]

                # some sessions starts with the system
                if dialog['utters'][0]['speaker'] == "System":
                    dialog_act = dialog_act[2::2]
                    dialog_info = dialog_info[1:]
                else:
                    dialog_act = dialog_act[1::2]

                cur_dom = []

                for t in dialog_act:
                    keys = [k['act'].lower() for k in t]
                    keys = ''.join(keys)
                    for d in doms:
                        if d in keys:
                            cur_dom.append(d)
                            break

                # dialog_meta = [t['metadata'] for t in dialog['log']]
                # dialog_meta = dialog_meta[1::2]
                # cs = convert_meta(dialog_meta, cur_dom)

                dp, sn = convert_act(dialog_act)
                sns = sns.union(sn)
                dialog_len = len(dialog_info)

                if dialog_len == 0:
                    continue
                utterances = {"utterances": []}
                temp = {"candidates": [], "history": [], "dp": [], "cs": []}

                for i in range(dialog_len):
                    if i % 2 == 0:
                        temp["history"].append(dialog_info[i])
                        temp["candidates"].append(random_candidates(data))
                        try:
                            temp["candidates"].append(dialog_info[i + 1])
                        except:
                            temp["candidates"].append(" ")
                        try:
                            temp["dp"].append(' '.join(dp[i // 2]))
                        except:
                            temp["dp"].append(' '.join(""))

                        """
                        if cs[i // 2] != '':
                            temp["cs"].append(cs[i // 2])"""

                    else:
                        utterances["utterances"].append(copy.deepcopy(temp))
                        temp["history"].append(dialog_info[i])
                        temp["candidates"] = []
                        temp["dp"] = []
                        # temp["cs"] = []
                dataset.append(utterances)
            print(list(sns))
            return dataset

        train = parse_woz_data(train_dataset)
        valid = parse_woz_data(valid_dataset)
        dataset = {"train": train, "valid": valid}
        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        # print("INPUT CONVERT: ")
        # print(dataset)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)

    return dataset


def download_model_from_googledrive(file_id, dest_path):

    gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, unzip=True)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == "__main__":

    download_model_from_googledrive('1FyL8nh3LmRIsWsYDR9pZvr0EKpWFrJ2G', './test')
