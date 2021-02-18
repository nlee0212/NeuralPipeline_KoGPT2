import json

requestable=set()
doms = ['contact','schedule','weather']
domain_request = {'contact':set(),'schedule':set(),'weather':set()}
acts = set()
slots = set()

def find_requestable(data):
    for session in data:
        for utter in session['utters']:
            if len(utter['requested']) != 0:
                for request in utter['requested']:
                    requestable.add(request)
                    if 'topic' not in utter.keys():
                        # print(utter['dialog_acts'][0]['act'])
                        domain_request['contact'].add(request)
                    else:
                        cur_dom = utter['topic'].lower()
                        if cur_dom not in doms:
                            cur_dom = ((utter['dialog_acts'][0]['act']).split('-'))[0]
                        # print(cur_dom)
                        domain_request[cur_dom].add(request)
            for dact in utter['dialog_acts']:
                if dact['act'] != None:
                    acts.add('<'+dact['act']+'>')
                if dact['slot'] != None:
                    if dact['slot'] != "None":
                        slots.add('<'+dact['slot']+'>')

with open("korean_train.json", "r", encoding='utf-8') as f:
    json_data = json.load(f)
    find_requestable(json_data)

with open("korean_valid.json", "r", encoding='utf-8') as f:
    json_data = json.load(f)
    find_requestable(json_data)

with open("korean_test.json", "r", encoding='utf-8') as f:
    json_data = json.load(f)
    find_requestable(json_data)

with open("requestables.txt","w",encoding='utf-8') as f:
    for request in requestable:
        f.write(request+" ")

for dom in doms:
    domain_request[dom] = list(domain_request[dom])

with open("domain-requestables.json","w",encoding='utf-8') as f:
    json.dump(domain_request,f,indent="\t")

with open("slot_list.txt","w",encoding='utf-8') as f:
    for slot in slots:
        f.write(slot+" ")

with open("act_list.txt","w",encoding='utf-8') as f:
    for act in acts:
        f.write(act+" ")

print(requestable)
print(domain_request)
print(acts)
print(slots)