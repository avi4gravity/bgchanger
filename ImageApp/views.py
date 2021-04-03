from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.http import HttpResponse,JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.http import HttpResponse
from .models import Images,Backgrounds,Images_DB
from django.conf import settings
import os 
import pixellib
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
            data={'MSG':'Success','image_url':'media/output/'+unique_filename,'img_data':image_data}
            
            
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
def two_bg_change(request):
    mask_fg=''
    data={}
    if request.method == 'POST':
        if len(request.FILES) !=0:
            
            fg = request.FILES.get('fg')
            bg = request.FILES.get('bg')
            fg_image = Images_DB.objects.create(image = fg)
            bg_image = Images_DB.objects.create(image = bg)
            # print(os.path.basename(str(fg_image.image)))
            fg=os.path.join(path,'uploads',os.path.basename(str(fg_image.image)))
            bg=os.path.join(path,'uploads',os.path.basename(str(bg_image.image)))
            print(fg)
            print(bg)
            data={'a':change_bg_2(fg,bg)}
            
        return JsonResponse(data)
        # bg=os.path.basename(bg)
    else: 
        print('Hhahha')
    return render(request,'ImageApp/upload1.html')

    
def home(request):
    return render(request,'ImageApp/home.html',{})


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
