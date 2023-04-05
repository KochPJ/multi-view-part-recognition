import json

with open('./results/logiccube_dict.json','r') as f:
    data = json.load(f)

    for cls, val in data['train'].items():
        for sample in val:
            for view in sample.values():
                temp = view['rgb']
                if ".png/" in temp:
                    view['rgb'] = temp[:-1]

with open('./results/logiccube_dict_new.json', 'w') as out_f:

    json_file= json.dump(data)
    #
    out_f.write(json_file)
