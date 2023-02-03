
import os
import json


if __name__ == '__main__':


    errors = []
    super_cls_lut = {'Generator': ['CarComponent'],
                     'Starter': ['CarComponent'],
                     'Tongs': ['Tool'],
                     'Screw': ['MetalPart'],
                     'Screwdriver': ['Tool'],
                     'CommonRailInjector': ['CarComponent'],
                     'Caliper': ['CarComponent'],
                     'Servopump': ['CarComponent'],
                     'MassAirFlowMeter': ['CarComponent'],
                     'Wrench': ['Tool'],
                     'Drill': ['Tool'],
                     }

    nl_lut = {
        'Iro': 'Iron',
        'Wood': 'Plastik',
        'silver': 'Grey',
        'no weight': 'Light',
        'Edges': 'Edgy',
        'select': 'Small',
        'high': 'High',
        'CarComponents': 'CarComponent'
    }

    samples = {}
    des = {}
    sup = {}
    root = './MVIP/sets'
    for cls in os.listdir(root):
        p = os.path.join(root, cls, 'train_data', '0', '0')

        sample = {'v': {},
                  'meta': os.path.join(root, cls, 'meta.json')}
        for cam in os.listdir(p):
            cam_id = cam.split('_')[-1]
            sample['v'][cam] = os.path.join(p, cam, '{}_rgb.png'.format(cam_id))
        samples[cls] = sample

        with open(sample['meta']) as f:
            data = json.load(f)

        if not isinstance(data['description'], dict):
            print('missing descriptions in ', cls, data['description'])
            data['description'] = {}

        new_des = {}
        for d, v in data['description'].items():
            if d == 'SuperClasses':
                continue

            if d not in des:
                des[d] = {}

            if d not in new_des:
                new_des[d] = []

            if not isinstance(v, list):
                v = [v]

            for k in v:
                if k == 'select':
                    print(cls, data)

                if k in nl_lut:
                    k = nl_lut[k]

                if k not in des[d]:
                    des[d][k] = 1 #.append(k)
                else:
                    des[d][k] += 1

                if k not in new_des[d]:
                    new_des[d].append(k)

        data['description'] = new_des
        subs = []
        for k in data.get('super_class', []):
            if k in nl_lut:
                k = nl_lut[k]
            if k in super_cls_lut:
                for s in super_cls_lut[k]:
                    if s not in subs:
                        subs.append(s)
            if k not in subs:
                subs.append(k)
        if len(subs) == 0:
            errors.append((cls, 'Missing superclass', data))
            #print('Missing superclass for {}'.format(cls))
            #print(data)
            #input()
        data['super_class'] = subs

        for k in data.get('super_class', []):
            if k not in sup:
                sup[k] = 1
            else:
                sup[k] += 1

        #with open(sample['meta'], 'w') as f:
        #    json.dump(data, f)

    #print('sup', sup)
    print('#### Superclasses #####')
    for k, v in sup.items():
        print(k, v)


    #print('des', des)

    print('#### Descriptions #####')
    for k, v in des.items():
        print(k)
        for k2, v2 in v.items():
            print('     {}: {}'.format(k2, v2))

    sample_id = 0


    keys = sorted(list(samples.keys()))
    for cls, error, data in errors:
        sampleid = keys.index(cls)
        print('ID : {}'.format(sampleid), cls, error, data)
