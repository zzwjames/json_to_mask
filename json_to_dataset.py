import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from melabel import *
from labelme import utils


if __name__ == '__main__':
    #jpgs_path   = "datasets/JPEGImages"                 #
    pngs_path   = "datasets/SegmentationClass"          #mask图片保存位置
    #classes     = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    classes     = ["_background_","rujiaoyiwu","fushi"]        #类别
    
    count = os.listdir("./datasets/before/")            #原图片以及json
    for i in range(0, len(count)):                      #循环处理每张图片
        path = os.path.join("./datasets/before", count[i])    #路径

        if os.path.isfile(path) and path.endswith('json'):    #json文件
            data = json.load(open(path))                      #加载json文件
            #"imageData": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQg
            '''if data['imageData']:
                imageData = data['imageData']
            else:                                                                      
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])   
                with open(imagePath, 'rb') as f:     #以二进制方式只读打开
                    imageData = f.read()             #读取所有字节   b'\xff\xd8\xff
                    imageData = base64.b64encode(imageData).decode('utf-8')  #编码、解码 /9j/4AAQSk
            #img_b64_to_arr将imagedata中的字符转化成原始图像
            img = utils.img_b64_to_arr(imageData)'''     
            label_name_to_value = {'_background_': 0}
            #给每个class赋值一个label
            for shape in data['step_1']['result']:
                label_name = shape['attribute']
                #if label_name=='rujiao' and 'rujiaoyiwu' in label_name_to_value:
                #    label_value = label_name_to_value['rujiaoyiwu']
                #if label_name=='yiwu' and 'rujiaoyiwu' in label_name_to_value:
                #    label_value = label_name_to_value['rujiaoyiwu']
                #else:
                if label_name=='fushi':
                    label_name_to_value['fushi'] = 2
                else: 
                    label_name_to_value['rujiaoyiwu']=1
            #print(label_name_to_value)
            
            # label_values must be dense
            #label_name_to_value:{background:0,cat:1,dog:2}
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]): #按照key的值排序
                label_values.append(lv)   #0,1,2
                label_names.append(ln)    #background,cat,dog
            #print(label_values)
            #print(label_names)
            assert label_values == list(range(len(label_values)))    #判断是否异常
            
            lbl = shapes_to_label((data['height'],data['width']), data['step_1']['result'], label_name_to_value)    
            #()放到JPEGImages文件夹里
            #PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))
            #np.set_printoptions(threshold=np.inf)
            #print(np.array(lbl))
            #new = np.zeros([data['height'],data['width']])
            #for name in label_names:
            #    index_json = label_names.index(name)
            #    index_all = classes.index(name)
                #给图像像素附上对应class的label(语义分割的图像)
            #    new = new + index_all*(np.array(lbl) == index_json)
            #print('this is new')
            #print(np.array(new))
            #PIL.Image.fromarray(lbl).save(osp.join(jpgs_path, count[i].split(".")[0]+'.png'))
            #在不改变像素值的情况下，给原图像加一层调色盘
            #utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), lbl)
            lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')     #uint8,8位无符号整形，mode和type对应
            lbl_pil.save(osp.join(pngs_path, count[i].split(".")[0]+'.png'))
            #print('Saved ')
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '_mask'+'.png')
