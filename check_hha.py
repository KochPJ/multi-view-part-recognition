import os
if __name__ == '__main__':
    path = '/mnt/share/more_datasets/MVIP'


    hha = 0
    rgb = 0
    total = 0
    train = 0
    valid = 0
    test = 0
    background = 0
    meta = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            if 'rgb.png' in file:
                rgb += 1
            elif 'hha.png' in file:
                hha += 1

            total += 1
            #print(file)
            if '.png' in file:
                if 'train' in root:
                    train += 1
                elif 'valid' in root:
                    valid += 1
                elif 'test' in root:
                    test += 1
                elif 'background' in root and '_rgb.png' in file:
                    print(file)
                    background += 1
            else:
                meta += 1

    print(hha, rgb, hha/rgb * 100)
    print('total', total)
    print('train', train)
    print('valid', valid)
    print('test', test)
    print('png', valid + train + test)
    print('background', background)
    print('meta', meta)
    print('rgb', rgb)
    print('rgb', rgb)