import json
from PIL import Image

with open('./results/logiccube_dict3.json','r') as f:
    data = json.load(f)

    for cls, val in data['train'].items():
        if len(val) < 3:
            print(cls + ' ' )
            print(len(val))
            for sample in val:
                if len(sample.values()) < 9:
                    print(cls + ' ')
                    print(len(sample.values()))


        # for sample in val:
            # if len(sample.values()) < 4:
            #   print(cls + " # " + sample.key())
            # print(len(sample.values()))

            # for view in sample.values():
            #     temp = view['rgb']
            #     if "logicNas" in temp:
            #         view['rgb'] = temp[:-1]
    samples = {}
    for cls, val in data['valid'].items():

        for sample in val:
            for view in sample.values():
                try:
                    img = Image.open(view['rgb'])
                    samples
                except:
                    print('Could not load Image ' + view['rgb'])
            if len(sample.values()) < 9:
                print(cls + ' ')
                print(len(sample.values()))
        if len(val) < 1:
            print(cls + ' ')
            print(len(val))




# with open('./results/logiccube_dict_new.json', 'w') as out_f:
#
#     json_file= json.dump(data)
#     #
#     out_f.write(json_file)
