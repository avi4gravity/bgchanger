from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.http import HttpResponse,JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.http import HttpResponse
from .models import Images,Backgrounds,Images_DB
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf
import os
import cv2
import numpy as np
from easydict import EasyDict as edict
from yaml import load

import matplotlib
import matplotlib.pyplot as plt
import pylab

import os
pwd=os.path.dirname(__file__)
import sys
sys.path.insert(0,pwd+'/PortraitNet/model/')
sys.path.insert(0,pwd+'/PortraitNet/data/')
print(sys.path)
def run_tflite_model(tflite_file, test_image):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  # Get input and output details
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # Preprocess the input image
  test_image = test_image/255.0
  test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])

  # Run the interpreter and get the output
  interpreter.set_tensor(input_details["index"], test_image)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details["index"])[0]

  # Compute mask from segmentaion output
  mask = np.reshape(output, (512,512))>0.5

  return mask
def mask_tf(image):
  image1=image
  print(image1.shape)
  image=cv2.resize(image,(512,512))
  mask=run_tflite_model(tflite_file=pwd+'/slim_reshape_v2.tflite',test_image=image)
  mask=mask.astype(np.uint8)
  mask=cv2.resize(mask,(image1.shape[1],image1.shape[0]))
  return mask
def chang_bg_a3(image_fg,image_bg):
    try:
        output_path=os.path.join(path,'output')
        img_path=os.path.join(path,'uploads')
        image= cv2.imread(os.path.join(img_path,image_fg))
        green=cv2.imread(image_bg)
        green=cv2.resize(green,(image.shape[1],image.shape[0]))
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask=mask_tf(image)
        trimap = np.zeros((mask.shape[0], mask.shape[1], 2))
        trimap[:, :, 1] = mask > 0
        trimap[:, :, 0] = mask == 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
        trimap[:, :, 0] = cv2.erode(trimap[:, :, 0], kernel)
        trimap[:, :, 1] = cv2.erode(trimap[:, :, 1], kernel)
        fg, bg, alpha = pred((image/255.0)[:, :, ::-1], trimap, model)
        blend = fg*alpha[:,:,None]*255.0 + green*(1 - alpha[:,:,None])
        unique_filename = str(uuid.uuid4())+'.jpg'
        blend=blend.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path,unique_filename),blend)
        image_data=image2base64(os.path.join(output_path,unique_filename))
        os.remove(os.path.join(output_path,unique_filename))
        os.remove(os.path.join(img_path,image_fg))
        os.remove(image_bg)
        data={'MSG':'Success','img_data':image_data}
        return data
    except Exception as e:
        image_data=image2base64(os.path.join(path,'150.png'))
        data={'MSG':e,'image_url':'abc','img_data':image_data}
        return data

    
from data_aug import Normalize_Img, Anti_Normalize_Img
def padding_img(img_ori, size=224, color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]
    img = np.zeros((max(height, width), max(height, width), 3)) + color
    
    if (height > width):
        padding = int((height-width)//2)
        img[:, padding:padding+width, :] = img_ori
    else:
        padding = int((width-height)//2)
        img[padding:padding+height, :, :] = img_ori
        
    img = np.uint8(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.array(img, dtype=np.float32)

def resize_padding(image, dstshape, padValue=0):
    height, width, _ = image.shape
    ratio = float(width)/height # ratio = (width:height)
    dst_width = int(min(dstshape[1]*ratio, dstshape[0]))
    dst_height = int(min(dstshape[0]//ratio, dstshape[1]))
    origin = [int((dstshape[1] - dst_height)//2), int((dstshape[0] - dst_width)//2)]
    if len(image.shape)==3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        newimage = np.zeros(shape = (dstshape[1], dstshape[0], image.shape[2]), dtype = np.uint8) + padValue
        newimage[origin[0]:origin[0]+dst_height, origin[1]:origin[1]+dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height),  interpolation = cv2.INTER_NEAREST)
        newimage = np.zeros(shape = (dstshape[1], dstshape[0]), dtype = np.uint8)
        newimage[origin[0]:origin[0]+height, origin[1]:origin[1]+width] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    return newimage, bbx

def generate_input(exp_args, inputs, prior=None):
    inputs_norm = Normalize_Img(inputs, scale=exp_args.img_scale, mean=exp_args.img_mean, val=exp_args.img_val)
    
    if exp_args.video == True:
        if prior is None:
            prior = np.zeros((exp_args.input_height, exp_args.input_width, 1))
            inputs_norm = np.c_[inputs_norm, prior]
        else:
            prior = prior.reshape(exp_args.input_height, exp_args.input_width, 1)
            inputs_norm = np.c_[inputs_norm, prior]
       
    inputs = np.transpose(inputs_norm, (2, 0, 1))
    return np.array(inputs, dtype=np.float32)

def pred_single(model, exp_args, img_ori, prior=None):
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    in_shape = img_ori.shape
    img, bbx = resize_padding(img_ori, [exp_args.input_height, exp_args.input_width], padValue=exp_args.padding_color)
    
    in_ = generate_input(exp_args, img, prior)
    in_ = in_[np.newaxis, :, :, :]
    
    if exp_args.addEdge == True:
        output_mask, output_edge = model(Variable(torch.from_numpy(in_)))
    else:
        output_mask = model(Variable(torch.from_numpy(in_)))
    prob = softmax(output_mask)
    pred = prob.data.cpu().numpy()
    
    predimg = pred[0].transpose((1,2,0))[:,:,1]
    out = predimg[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    out = cv2.resize(out, (in_shape[1], in_shape[0]))
    return out, predimg
# config_path = 'PortraitNet/config/model_mobilenetv2_with_two_auxiliary_losses.yaml'

# load model-3: trained with prior channel
config_path = pwd+'/PortraitNet/config/model_mobilenetv2_with_prior_channel.yaml'
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

with open(config_path,'rb') as f:
    cont = f.read()
cf = load(cont)

print ('finish load config file ...')
exp_args = edict()    
exp_args.istrain = False
exp_args.task = cf['task']
exp_args.datasetlist = cf['datasetlist'] # ['EG1800', ATR', 'MscocoBackground', 'supervisely_face_easy']

exp_args.model_root = cf['model_root'] 
exp_args.data_root = cf['data_root']
exp_args.file_root = cf['file_root']

# the height of input images, default=224
exp_args.input_height = cf['input_height']
# the width of input images, default=224
exp_args.input_width = cf['input_width']

# if exp_args.video=True, add prior channel for input images, default=False
exp_args.video = cf['video']
# the probability to set empty prior channel, default=0.5
exp_args.prior_prob = cf['prior_prob']

# whether to add boundary auxiliary loss, default=False
exp_args.addEdge = cf['addEdge']
# whether to add consistency constraint loss, default=False
exp_args.stability = cf['stability']

# input normalization parameters
exp_args.padding_color = cf['padding_color']
exp_args.img_scale = cf['img_scale']
# BGR order, image mean, default=[103.94, 116.78, 123.68]
exp_args.img_mean = cf['img_mean']
# BGR order, image val, default=[0.017, 0.017, 0.017]
exp_args.img_val = cf['img_val'] 

# if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
exp_args.useUpsample = cf['useUpsample'] 
# if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
exp_args.useDeconvGroup = cf['useDeconvGroup'] 
import model_mobilenetv2_seg_small as modellib
netmodel_video = modellib.MobileNetV2(n_class=2, 
                                      useUpsample=exp_args.useUpsample, 
                                      useDeconvGroup=exp_args.useDeconvGroup, 
                                      addEdge=exp_args.addEdge, 
                                      channelRatio=1.0, 
                                      minChannel=16, 
                                      weightInit=True,
                                      video=exp_args.video)


bestModelFile = pwd+'/Model/PortraitSegments.pth'
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
import sys
sys.path.append(pwd+"/FBA_Matting")

from demo import np_to_torch, pred, scale_input
from dataloader import read_image, read_trimap
from networks.models import build_model

netmodel_video.stage0[0].padding=(1,1)
netmodel_video.transit1.block[0][0].padding=(1,1)
netmodel_video.transit2.block[0][0].padding=(1,1)
netmodel_video.transit3.block[0][0].padding=(1,1)
netmodel_video.transit4.block[0][0].padding=(1,1)
netmodel_video.transit5.block[0][0].padding=(1,1)

print('Model Made')
checkpoint_video = torch.load(bestModelFile)
netmodel_video.load_state_dict(checkpoint_video)
class Args:
    encoder = 'resnet50_GN_WS'
    decoder = 'fba_decoder'
    weights = pwd+'/Model/FBA.pth'

args=Args()

model = build_model(args)


from django.conf import settings
import os 
# from pixellib.semantic import semantic_segmentation
import PIL
from PIL import Image
import numpy as np 
import cv2
import hashlib
import json
from django.http import HttpResponse
import matplotlib.pyplot as plt
from copy import deepcopy
from django.http import JsonResponse
# segment_image = semantic_segmentation()
path=(settings.MEDIA_ROOT)
# segment_image.load_pascalvoc_model(os.path.join(path,"deeplabv3_xception_tf_dim_ordering_tf_kernels_improved.h5"))
import random
import base64
import uuid

import sqlite3
def image2base64(img):
    with open(img, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    return image_data
conn = sqlite3.connect('test.db')
try:
    conn.execute('''CREATE TABLE IMG_DB
         (ID TEXT PRIMARY KEY   NOT NULL,
         MASK            TEXT     NOT NULL);''')
    print('Table Created')
except Exception as e:
    print('Table Exists',e)
    pass
def check_mask_exists(fg):
    print(fg)
    conn = sqlite3.connect('test.db')
    file_hash=hashlib.md5(open(fg,'rb').read()).hexdigest()
    print(file_hash)
    cursor = conn.execute("SELECT MASK from IMG_DB where ID='"+file_hash+"'")
    x=[i[0] for i in cursor]
    conn.close()
    if not x:
        print('Mask does not exists')
        return None
    else:
        file_mask=os.path.join(path,'masks',x[0])
        
        if os.path.isfile(file_mask):
            return file_mask
        else:
            print('mask file not found')
            return None
    

# Create your views here.
def make_mask2(img_name):
    try:
#         file_name=os.path.basename(img_name)
#         img_path=os.path.join(path,'uploads')
#         mask_path=os.path.join(path,'masks')
#         #a=segment_image.segmentAsPascalvoc(os.path.join(img_path,file_name), output_image_name = os.path.join(mask_path,file_name))
#         img=cv2.imread(os.path.join(img_path,file_name))
#         mask=cv2.imread(os.path.join(mask_path,file_name),0)
#         (thresh, mask) = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         kernel = np.ones((3,3), np.uint8)
#         mask = cv2.erode(mask,kernel,iterations = 3)
#         cv2.imwrite(os.path.join(mask_path,file_name),mask)
        return {'msg':'Successful','file':file_name}
    except Exception as e:
        print(e)
        return {'msg':e,'file':None}
def make_mask(img_name):
#     try:
#         file_name=os.path.basename(img_name)
#         img_path=os.path.join(path,'uploads')
#         mask_path=os.path.join(path,'masks')
#         a=segment_image.segmentAsPascalvoc(os.path.join(img_path,file_name), output_image_name = os.path.join(mask_path,file_name))
#         img=cv2.imread(os.path.join(img_path,file_name))
#         mask=cv2.imread(os.path.join(mask_path,file_name),0)
#         (thresh, mask) = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         kernel = np.ones((3,3), np.uint8)
#         mask = cv2.erode(mask,kernel,iterations = 3)
#         cv2.imwrite(os.path.join(mask_path,file_name),mask)
#         return 'Successful'
#     except Exception as e:
#         print(e)
#         return e
        return 'error'
def make_shadow(fg,bg,mask):
    try:
        img=fg
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        extended=cv2.drawContours(np.zeros((img.shape)), contours, -1, (169,169,169), -1,offset=(-4,-4))
        extended1=cv2.drawContours(np.zeros((img.shape[0],img.shape[1])), contours, -1, (255,255,255), -1,offset=(-4,-4))
        fg_1=np.ones(fg.shape)*192
        extended=extended.astype(dtype='uint8')
        res_fg=cv2.bitwise_and(fg_1,fg_1,mask = extended1.astype('uint8'))
        blur = cv2.GaussianBlur(res_fg,(11,11),0)
        res_bg=cv2.bitwise_and(bg,bg,mask = np.bitwise_not(extended1.astype('uint8')))
        blur=blur.astype('uint8')
        dst = cv2.addWeighted(bg,1,blur,-0.3,0)
        output_path=os.path.join(path,'output')
        unique_filename = str(uuid.uuid4())+'.jpg'
        cv2.imwrite(os.path.join(output_path,unique_filename),dst)
        return dst
    except Exception as e:
        print(e)
        return bg

def change_bg_2(fg,bg):
    
    mask_fg=''
    try:
        msg_1=check_mask_exists(fg)
        msg={}
        if msg_1:
            mask_fg=msg_1
            print(mask_fg)
            msg['msg']='Successful'
        else:
#             msg=make_mask2(fg)
          
            file_hash=hashlib.md5(open(fg,'rb').read()).hexdigest()
            mask_fg=msg['file']
            try:
                conn = sqlite3.connect('test.db')
                conn.execute("INSERT INTO IMG_DB (ID,MASK) VALUES ('"+file_hash+"','"+mask_fg+"')");
                conn.commit()
                conn.close()
            except Exception as e:
                print(e)
                msg['msg']=e
            
            
        # bg=os.path.basename(bg)
        print(msg)
        if msg['msg']=='Successful':
            img_path=os.path.join(path,'uploads')
            mask_path=os.path.join(path,'masks')
            # bg_path=os.path.join(path_,'background_pics')
            output_path=os.path.join(path,'output')
            
            print(img_path,mask_fg,bg,output_path)
            mask=cv2.imread(os.path.join(mask_path,mask_fg),0)
            (thresh, mask) = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            fg=cv2.imread(os.path.join(img_path,fg))
            bg=cv2.imread(bg)
            bg=cv2.resize(bg,(fg.shape[1],fg.shape[0]))
            bg=make_shadow(fg,bg,mask)
            res_fg=cv2.bitwise_and(fg,fg,mask = mask)
            res_bg = cv2.bitwise_and(bg,bg,mask = cv2.bitwise_not(mask))
            new_image=res_fg+res_bg
            unique_filename = str(uuid.uuid4())+'.jpg'
            cv2.imwrite(os.path.join(output_path,unique_filename),new_image)
            # domain = request.get_host()
            image_data=image2base64(os.path.join(output_path,unique_filename))
            os.remove(os.path.join(output_path,unique_filename))
            
            data={'MSG':'Success','img_data':image_data}
            
            
            return data
        else:
            image_data=image2base64(os.path.join(path,'150.png'))
            data={'MSG':'Error','image_url':fg,'img_data':image_data}
            
            print('Some Error Occurs')
           
            return data
    except Exception as e:
        
        image_data=image2base64(os.path.join(path,'150.png'))
        data={'MSG':e,'image_url':fg,'img_data':image_data}
            
       
        return data
def change_bg_a1(image_fg,image_bg):
    try:
        output_path=os.path.join(path,'output')
        img_path=os.path.join(path,'uploads')
        img_ori = cv2.imread(os.path.join(img_path,image_fg))
        green=cv2.imread(image_bg)
        print(img_ori.shape)
        green=cv2.resize(green,(img_ori.shape[1],img_ori.shape[0]))
        prior = None
        height, width, _ = img_ori.shape
        alphargb, pred1 = pred_single(netmodel_video, exp_args, img_ori, prior)
        mask=alphargb>0.7
        print(mask.shape)
        print(green.shape)
        trimap = np.zeros((mask.shape[0], mask.shape[1], 2))
        trimap[:, :, 1] = mask > 0
        trimap[:, :, 0] = mask == 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
        trimap[:, :, 0] = cv2.erode(trimap[:, :, 0], kernel)
        trimap[:, :, 1] = cv2.erode(trimap[:, :, 1], kernel)
        fg, bg, alpha = pred((img_ori/255.0)[:, :, ::-1], trimap, model)
        blend =cv2.cvtColor((fg*alpha[:,:,None])*255.0,cv2.COLOR_BGR2RGB) + green*(1 - alpha[:,:,None])
        unique_filename = str(uuid.uuid4())+'.jpg'
        blend=blend.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path,unique_filename),blend)
        del blend,fg,bg,alpha,trimap,mask,alphargb,green,img_ori,pred1,
        image_data=image2base64(os.path.join(output_path,unique_filename))
        os.remove(os.path.join(output_path,unique_filename))
        os.remove(image_bg)
        os.remove(os.path.join(img_path,image_fg))

        data={'MSG':'Success','img_data':image_data}
        return data
    except Exception as e:
        image_data=image2base64(os.path.join(path,'150.png'))
        data={'MSG':e,'image_url':'abc','img_data':image_data}
        return data
    
def two_bg_change(request):
    mask_fg=''
    data={}
    if request.method == 'POST':
        if len(request.FILES) !=0:
            model_s=request.POST.get('model')
            model_s=int(model_s)
            fg = request.FILES.get('fg')
            bg = request.FILES.get('bg')
            fg_image = Images_DB.objects.create(image = fg)
            bg_image = Images_DB.objects.create(image = bg)
            # print(os.path.basename(str(fg_image.image)))
            fg=os.path.join(path,'uploads',os.path.basename(str(fg_image.image)))
            bg=os.path.join(path,'uploads',os.path.basename(str(bg_image.image)))
            print(fg)
            print(bg)
            if model_s==1:
                data={'a':change_bg_a1(fg,bg)}
            elif model_s==2:
                data={'a':chang_bg_a3(fg,bg)}
        
            
        return JsonResponse(data)
        # bg=os.path.basename(bg)
    else: 
        print('Hhahha')
    return render(request,'ImageApp/upload1.html')



def home(request):
    return render(request,'ImageApp/home.html',{})

def change_bg_view(request):
    return 'abc'
'''
@login_required
def UploadView(request):
    return render(request, 'ImageApp/upload.html', {})
'''
@login_required
def ImagesView(request):
    data = Images.objects.all().filter(user_name=request.user)
    if len(data)!=0:
        context = {
            'images_data' : data,
            }
        return render(request, 'ImageApp/images.html', context)
    else:
        return render(request, 'ImageApp/images.html', {})


def ImageBackChanger(fg_image,bg_image):
    print(fg_image,bg_image)
    
    print(os.path.join(path, os.path.abspath(fg_image)))

    # Your AI code will go here.
    # Here you have user image primary key and background image primary key.
    # retrieve them from database.
    print("Integrate your code here please")

@login_required
def file_upload_view(request):
    #print(len(request.FILES))
    back_imgs = Backgrounds.objects.all()
    if request.method == 'POST':
        if len(request.FILES) !=0:
            myfile = request.FILES.get('file')
            current_image = Images.objects.create(image = myfile,user_name=request.user)
            # back_img = Backgrounds.objects.filter(pk=backid).values()
            fg_name=current_image.image.name
            msg=make_mask(fg_name)
            if msg=='Successful':
                context = {
                'msg':'Success',
                'back_images_data' : back_imgs,
                'uploaded_data': current_image
                }
                messages.success(request, f'Your File Uploaded',extra_tags='danger')
            else:
                context = {
                'msg':'Error',
                'back_images_data' : back_imgs,
                'uploaded_data': current_image
                }
                messages.error(request, msg,extra_tags='danger')
            # print(f'Current image :{current_image.pk} || then back image {back_name}')
            print(current_image)
            
           
            return render(request,'ImageApp/upload.html', context=context)
        else:
            messages.error(request, f'You must submit the file to upload!',extra_tags='danger')
            return render(request,'ImageApp/nofile.html',{})
    back_imgs = Backgrounds.objects.all()
    try:
        input_image=Images.objects.filter(user_name=request.user.id).latest('id')
        input_image=str(input_image.image)
    except:
        input_image='150.png'
    context = {
        'back_images_data' : back_imgs,
        'input_image':input_image
    }
    return render(request,'ImageApp/upload.html', context)

'''
@login_required
def background(request,imgID):
    back_imgs = Backgrounds.objects.all()
    preview_image = Images.objects.all().filter(pk=imgID)
    print(preview_image)
    if len(back_imgs)!=0:
        context = {
            'preview_image'    : preview_image,
            'back_images_data' : back_imgs
            }
        return render(request,'ImageApp/backgrounds.html', context)
    else:
        return render(request,'ImageApp/backgrounds.html', {})
'''
@login_required
def ImagesSelected(request,user_img,back_img):
    back_image = Backgrounds.objects.all().filter(pk=back_img)
    user_image = Images.objects.all().filter(pk=user_img)
    print("Im here")
    context = {
            'user_image' : user_image,
            'back_image' : back_image
            }
    return render(request,'ImageApp/selectedimages.html', context)
