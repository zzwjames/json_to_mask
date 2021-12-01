import math

import numpy as np
import PIL.Image
import PIL.ImageDraw


def shape_to_mask(img_shape,points,shape_type=None,line_width=10,point_size=5):
    mask=np.zeros(img_shape[:2],dtype=np.uint8)
    mask=PIL.Image.fromarray(mask)
    draw=PIL.ImageDraw.Draw(mask)
    #xy=[tuple(point) for point in points]
    #point_x=[tuple(point_x) for point_x in points['x']]
    #point_y=[tuple(point_y) for point_y in points['y']]
    #xy=tuple("{0}{1}".format(x,y) for x,y in zip(point_x,point_y))
    #xy=[tuple(int(point)) for point in points]
    print(points[0])
    point_x=tuple([point['x'] for point in points])
    point_y=tuple([point['y'] for point in points])
    #point_xy=tuple("{0},{1}".format(x,y) for x,y in zip(point_x,point_y))
    #xy=[tuple(point_x,point_y) for point_x,point_y in point_xy.split(str=",",num=1)]
    #xy=[tuple(point) for point in point_xy]
    xy=[]
    for i in range(len(point_x)):
        xy.append((point_x[i],point_y[i]))
    #print('point_x:')
    #print(point_x)
    #print('point_y:')
    #print(point_y)
    #print('xy:')
    #print(xy)
    #print('point_xy:')
    #print(point_xy)
    #print(xy)
    draw.polygon(xy=xy,outline=1,fill=1)
    mask=np.array(mask,dtype=bool)
    return mask





def shapes_to_label(img_shape,shapes,label_name_to_value):
    cls=np.zeros(img_shape[:2],dtype=np.int32)
    for shape in shapes:             #遍历一张图的多个物体
        points=shape['pointList']
        label=shape['attribute']
        shape_type='polygonTool'
        cls_name=label
        if cls_name=='rujiao' or cls_name=='yiwu':
            cls_id=1
        else:
            cls_id=2  #统一各张图class对应的索引
        mask=shape_to_mask(img_shape[:2],points,shape_type)
        cls[mask]=cls_id
    return cls
