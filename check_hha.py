import os
if __name__ == '__main__':
    path = '/mnt/share/more_datasets/MVIP/sets'


    hha = 0
    rgb = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'rgb.png' in file:
                rgb += 1
            elif 'hha.png' in file:
                hha += 1

    print(hha, rgb, hha/rgb * 100)