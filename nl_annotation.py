import os
import json
import tkinter as tk
#from tkinter import *
from ttkbootstrap import Style
#from tkinter import ttk, Canvas
from PIL import Image, ImageTk
#from tkinter import


#
#meta_data = 0

# Create A Main frame
#main_frame = tk.Frame(root, bg="white")
#main_frame.pack(fill=tk.BOTH, expand=1)

win = tk.Tk()
#style = Style(theme="lumen")
#root = style.master
height = 1080
width = 1920
win.title('MVIP')
win.geometry("%dx%d" % (width, height))
#root.minsize(width, height)


#frame = tk.Frame(win, width=width, height=height)
#frame.pack()
#frame.place(anchor='center', relx=0.5, rely=0.5)

# Create an object of tkinter ImageTk
#my_canvas = tk.Canvas(root, width=width, height=height)
#my_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


# Create Frame for X Scrollbar
#sec = Frame(main_frame, bg="white")
#sec.pack(fill=X, side=BOTTOM)

# Create A Canvas
'''

x_scrollbar = ttk.Scrollbar(sec, orient=HORIZONTAL, command=my_canvas.xview)

x_scrollbar.pack(side=BOTTOM, fill=X)

y_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
y_scrollbar.pack(side=RIGHT, fill=Y)

# Configure the canvas
my_canvas.configure(xscrollcommand=x_scrollbar.set)

my_canvas.configure(yscrollcommand=y_scrollbar.set)

my_canvas.bind("<Configure>", lambda e: my_canvas.config(scrollregion=my_canvas.bbox(ALL)))
sub_frame = Frame(my_canvas, width=1100, height=1000, bg="white")
'''

samples = {}
des = {}
sup = []
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
    for d, v in data['description'].items():
        if d == 'SuperClasses':
            continue
        if d not in des:
            des[d] = []

        if not isinstance(v, list):
            v = [v]
        for k in v:
            if k not in des[d]:
                des[d].append(k)

    for k in data.get('super_class', []):
        if k not in sup:
            sup.append(k)

print('sup', sup)
print('des', des)
sample_id = 0

a = 0.9
b = 0.1
c = 0.9
hn = 5
wn = 2
imgh = 1400
imgw = 2500

h = height * a
#w = width * b

h0 = int((height - h) // 2)
w0 = h0

h_ = int((c / hn) * h)
w_ = int(imgw * (h_ / imgh))

H_ = (1/hn) * h

pos_lut = {
}
maxy = 0
for i in range(10):
    y = i % hn

    y = h0 + int(y * H_)
    x = i // hn

    x = w0 + x * (w_ + int(b * w_))
    pos_lut['cam_{}'.format(str(i+1).zfill(2))] = (x, y)
    if y > maxy:
        maxy = y


labels = {}
frames = {}
images = {}

but_h = 0.5
but_w = 0.1
bu = 0.95
buoff = 1.1


def next(save=False):
    global sample_id

    if save:
        save_meta()

    sample_id += 1
    if sample_id >= len(samples.keys()):
        sample_id = 0
    for widget in win.winfo_children():
        widget.destroy()

    set_view()


def previous():
    global sample_id
    sample_id -= 1
    if sample_id < 0:
        sample_id = len(samples.keys()) - 1
    for widget in win.winfo_children():
        widget.destroy()
    set_view()



menus = {}
vars = {}
dframes = {}

current_meta = {}
texts = {}
font = ('Arial', 25)
font_small = ('Arial', 16)


n_menus = len(des) + 1

def update_des(*args, **kwargs):
    global vars, current_meta, texts, dframes, sframes
    for d in vars:

        texts[d].destroy()
        if 'Super' in d:
            current_meta['super_class'] = []
            for key, v in vars[d].items():
                if v.get():
                    current_meta['super_class'].append(key)
            texts[d] = tk.Label(sframes[d + '_text'], text=str(current_meta['super_class']), font=font_small)
        else:
            current_meta['description'][d] = []
            for key, v in vars[d].items():
                if v.get():
                    current_meta['description'][d].append(key)

            texts[d] = tk.Label(dframes[d+'_text'], text=str(current_meta['description'][d]), font=font_small)
        texts[d].pack()

strvar = None
save_meta_flag = True

def save_meta():
    global current_meta, sample_id, samples
    print('save_meta_flag', save_meta_flag)
    if save_meta_flag:
        keys = sorted(list(samples.keys()))
        # print(len(keys))
        sample = samples[keys[sample_id]]

        #with open(sample['meta']) as f:
        #    current_meta['old'] = json.load(f)
        #if 'old' in current_meta:
        #    del current_meta['old']

        if 'SuperClasses' in current_meta['description']:
            del current_meta['description']['SuperClasses']

        with open(sample['meta'], 'w') as f:
            json.dump(current_meta, f)
        print('saved current meta {}: {}'.format(sample_id, current_meta))



def add_text(*args, **kwargs):
    global strvar, dframes, sup, current_meta, des, samples, sample_id, win

    d = strvar.get()
    s = str(args[0].widget.get('1.0', 'end-1c'))
    args[0].widget.delete('1.0', tk.END)

    if d == 'Sample ID':
        try:
            s = int(s)
        except:
            print('cant convert "{}" to int'.format(s))
            return None
        if s < 1:
            print('Sample id "{}" can not be less then 1'.format(s))
        elif s > len(samples):
            print('Sample id "{}" can exceed the number of samples {}'.format(s, len(samples)))
        else:
            sample_id = s - 1
            for widget in win.winfo_children():
                widget.destroy()
            set_view()
        return None

    if d == 'SuperClasses':
        sup.append(s)
        current_meta['super_class'].append(s)
    else:
        current_meta['description'][d].append(s)
        des[d].append(s)

    set_des()

sframes = {}
def set_des():
    global des, vars, dframes, current_meta, texts, sup, strvar, sframes

    start = 100
    offset = 80
    counter = 0

    d = 'add_to_selection'
    sframes[d] = tk.Frame(win, height=int(n_menus / (height * 0.9)), width=int((width - maxy) * 0.3))
    sframes[d].pack()
    sframes[d].place(x=int(maxy * 1.05), y=start + counter * offset)

    sframes[d + '_text'] = tk.Frame(win, height=int(n_menus / (height * 0.9)), width=2*offset)
    sframes[d + '_text'].pack()
    sframes[d + '_text'].place(x=int(maxy * 1.05) + int((width - maxy) * 0.3), y=start + counter * offset)

    strvar = tk.StringVar(win)
    strvar.set('SuperClasses')

    textdes = ['SuperClasses', 'Sample ID'] + list(des.keys())
    menus[d] = tk.OptionMenu(sframes[d], strvar, *textdes)
    menus[d].pack()

    texts['input_text'] = tk.Text(sframes[d + '_text'], height=int(n_menus / (height * 0.9)),
                                  width=2*offset)
    texts['input_text'].pack()
    texts['input_text'].bind('<Return>', add_text)

    # menus[d].place(x=maxy+int((width-maxy)*0.1), y=100 + 100+i)

    counter += 1

    d = 'SuperClasses'
    sframes[d] = tk.Frame(win, height=int(n_menus / (height * 0.9)), width=int((width - maxy) * 0.3))
    sframes[d].pack()
    sframes[d].place(x=int(maxy * 1.05), y=start + counter * offset)

    sframes[d + '_text'] = tk.Frame(win, height=int(n_menus / (height * 0.9)), width=int((width - maxy) * 0.6))
    sframes[d + '_text'].pack()
    sframes[d + '_text'].place(x=int(maxy * 1.05) + int((width - maxy) * 0.3), y=start + counter * offset)

    menus[d] = tk.Menubutton(sframes[d], text='Sel. {}'.format(d), relief=tk.RAISED, font=font)
    # menus[d].place(x=maxy+int((width-maxy)*0.1), y=100 + 100+i)
    menus[d].menu = tk.Menu(menus[d], tearoff=0)
    menus[d]['menu'] = menus[d].menu

    if d not in vars:
        vars[d] = {}

    textdes = []
    for v in sorted(sup):
        if v in current_meta['super_class']:
            k = True
            textdes.append(v)
        else:
            k = False

        vars[d][v] = tk.BooleanVar(value=k)
        vars[d][v].trace('w', update_des)
        menus[d].menu.add_checkbutton(label=v, variable=vars[d][v], font=font)
    menus[d].pack()
    texts[d] = tk.Label(sframes[d + '_text'], text=str(current_meta['super_class']), font=font_small)
    texts[d].pack()
    counter += 1

    for i, (d, vs) in enumerate(des.items()):
        #print(int(n_menus/(height*0.9)), int((width-maxy)*0.9), maxy+int((width-maxy)*0.1), 100 + 100*i)

        dframes[d] = tk.Frame(win, height=int(n_menus/(height*0.9)), width=int((width-maxy)*0.3))
        dframes[d].pack()
        dframes[d].place(x=int(maxy * 1.05), y=start + counter * offset)

        dframes[d+'_text'] = tk.Frame(win, height=int(n_menus/(height*0.9)), width=int((width-maxy)*0.6))
        dframes[d+'_text'].pack()
        dframes[d+'_text'].place(x=int(maxy * 1.05) + int((width-maxy)*0.3), y=start + counter * offset)


        menus[d] = tk.Menubutton(dframes[d], text='Sel. {}'.format(d), relief=tk.RAISED, font=font)
        #menus[d].place(x=maxy+int((width-maxy)*0.1), y=100 + 100+i)
        menus[d].menu = tk.Menu(menus[d], tearoff=0)
        menus[d]['menu'] = menus[d].menu
        if d not in vars:
            vars[d] = {}

        if not isinstance(vs, list):
            vs = [vs]

        textdes = []
        for v in sorted(vs):
            if v in current_meta['description'][d]:
                k = True
                textdes.append(v)
            else:
                k = False

            vars[d][v] = tk.BooleanVar(value=k)
            vars[d][v].trace('w', update_des)
            menus[d].menu.add_checkbutton(label=v, variable=vars[d][v], font=font)
        counter += 1

        texts[d] = tk.Label(dframes[d+'_text'], text=str(textdes), font=font_small)
        texts[d].pack()
        #dframes[d].pack()
        menus[d].pack()


    sframes['size'] = tk.Frame(win, height=int(n_menus / (height * 0.9)), width=int((width - maxy) * 0.3))
    sframes['size'].pack()
    sframes['size'].place(x=int(maxy * 1.05), y=start + counter * offset)

    sframes['size_lbl'] = tk.Label(sframes['size'],
                                   text='Size: {}mm, weight: {}kg'.format(current_meta.get('size'),
                                                                      current_meta.get('weight')), font=font)
    sframes['size_lbl'].pack()

buttons = {}

def set_view():
    global buttons
    frames['next'] = tk.Frame(win, width=width * but_w, height=height * but_h)
    frames['next'].pack()
    frames['next'].place(x=(int(width) // 2), y=height * bu)
    buttons['next'] = tk.Button(frames['next'], text="Next", padx=60, pady=20, command=lambda: next())
    buttons['next'].pack()

    frames['next&save'] = tk.Frame(win, width=width * but_w, height=height * but_h)
    frames['next&save'].pack()
    frames['next&save'].place(x=(int(width) // 2) + int(width * but_w * buoff), y=height * bu)
    buttons['next&save'] = tk.Button(frames['next&save'], text="Save & Next", padx=60, pady=20, command=lambda: next(save=True))
    buttons['next&save'].pack()

    frames['previous'] = tk.Frame(win, width=width * but_w, height=height * but_h)
    frames['previous'].pack()
    frames['previous'].place(x=(int(width) // 2) - int(width * but_w * buoff), y=height * bu)
    buttons['previous'] = tk.Button(frames['previous'], text="Previous", padx=60, pady=20,
                                 command=lambda: previous())
    buttons['previous'].pack()


    keys = sorted(list(samples.keys()))
    #print(len(keys))
    sample = samples[keys[sample_id]]
    #print(sample)
    frames['sample_id'] = tk.Frame(win, width=int(maxy), height=int((((1-c) / 2) * 0.75) * height))
    frames['sample_id'].pack()
    frames['sample_id'].place(x=w0, y=int((((1-c) / 2) * 0.2) * height))
    labels['sample_id'] = tk.Label(frames['sample_id'], text='Sample {}/{}: {}'.format(
        sample_id+1, len(keys), keys[sample_id]), font=font)
    labels['sample_id'].pack()

    for cam, p in sample['v'].items():
        images[cam] = ImageTk.PhotoImage(Image.open(p).resize((w_, h_)))
        x, y = pos_lut[cam]
        frames[cam] = tk.Frame(win, width=w_, height=h_)
        frames[cam].pack()
        frames[cam].place(x=x, y=y)

        labels[cam] = tk.Label(frames[cam], image=images[cam])
        labels[cam].pack()
    
    global current_meta
    with open(sample['meta']) as f:
        current_meta = json.load(f)
    if 'description' not in current_meta:
        current_meta['description'] = {}
    if not isinstance(current_meta['description'], dict):
        current_meta['description'] = {}

    if 'SuperClasses' in current_meta['description']:
        del current_meta['description']['SuperClasses']
    for d in des:
        if d not in current_meta['description']:
            current_meta['description'][d] = []
    print('current_meta: {} from {}'.format(current_meta, sample['meta']))
    
    if 'super_class' not in current_meta:
        current_meta['super_class'] = []
    set_des()
    
        

def main():

    #img = ImageTk.PhotoImage(Image.open(path).resize((450, 300), Image.ANTIALIAS))
    #tk.Label(sub_frame, image=img, padx=60, pady=20).place(x=10, y=200)
    #ttk.Label(sub_frame, text="Sensorische Erfassung automatisierte Identifikation", foreground='#3498db',
    #          font=('sans-serif', 12, 'bold italic')).place(x=450, y=150)
    #Label(sub_frame, text="und Bewertung von Altteilen", foreground='#3498db', font=('sans-serif', 12, 'bold')).place(
    #    x=630, y=170)
    #Button(sub_frame, text="connect", padx=60, pady=20, command=connection).place(x=650, y=350)

    #ttk.Label(sub_frame, text="(connection will done with RealSence cameras)", foreground='#3498db',
    #          font=('sans-serif', 8)).place(x=610, y=430)
    set_view()
    win.mainloop()

def connection():
    pass

if __name__ == '__main__':
    main()








