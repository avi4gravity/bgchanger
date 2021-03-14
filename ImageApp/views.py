from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.http import HttpResponse,JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Images,Backgrounds
from django.conf import settings
import os 
import pixellib
from pixellib.semantic import semantic_segmentation
import PIL
from PIL import Image
import numpy as np 
import cv2
import json
from django.http import HttpResponse
import matplotlib.pyplot as plt
from copy import deepcopy
from django.http import JsonResponse
segment_image = semantic_segmentation()
path=(settings.MEDIA_ROOT)
segment_image.load_pascalvoc_model(os.path.join(path,"deeplabv3_xception_tf_dim_ordering_tf_kernels_improved.h5"))
import random

import uuid



# Create your views here.
def make_mask(img_name):
    try:
        file_name=os.path.basename(img_name)
        img_path=os.path.join(path,'uploads')
        mask_path=os.path.join(path,'masks')
        a=segment_image.segmentAsPascalvoc(os.path.join(img_path,file_name), output_image_name = os.path.join(mask_path,file_name))
        img=cv2.imread(os.path.join(img_path,file_name))
        mask=cv2.imread(os.path.join(mask_path,file_name),0)
        (thresh, mask) = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 3)
        cv2.imwrite(os.path.join(mask_path,file_name),mask)
        return 'Successful'
    except Exception as e:
        print(e)
        return e
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

def change_bg(request,fg,bg):
    try:
        output=fg=os.path.basename(fg)
        bg=os.path.basename(bg)
        img_path=os.path.join(path,'uploads')
        mask_path=os.path.join(path,'masks')
        bg_path=os.path.join(path,'background_pics')
        output_path=os.path.join(path,'output')
        print(img_path,mask_path,bg_path,output_path)
        mask=cv2.imread(os.path.join(mask_path,fg),0)
        (thresh, mask) = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        fg=cv2.imread(os.path.join(img_path,fg))
        bg=cv2.imread(os.path.join(bg_path,bg))
        bg=cv2.resize(bg,(fg.shape[1],fg.shape[0]))
        bg=make_shadow(fg,bg,mask)
        res_fg=cv2.bitwise_and(fg,fg,mask = mask)
        res_bg = cv2.bitwise_and(bg,bg,mask = cv2.bitwise_not(mask))
        new_image=res_fg+res_bg
        unique_filename = str(uuid.uuid4())+'.jpg'
        cv2.imwrite(os.path.join(output_path,unique_filename),new_image)
        domain = request.get_host()
        data={'MSG':'Success','image_url':'/media/output/'+unique_filename}
        return data
    except Exception as e:
        bg=os.path.basename(bg)
        data={'MSG':e,'image_url':'media/background_pics/'+output}
        return data
    
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
def change_bg_view(request):
    if request.method == 'POST':
        back_id=request.POST.get('back_id')
        back_image=Backgrounds.objects.get(id=back_id)
        back_image=str(back_image.back_img)
        front_image=Images.objects.filter(user_name=request.user.id).latest('id')
        front_image=str(front_image.image)
        msg=change_bg(request,front_image,back_image)
        return HttpResponse(json.dumps(msg), content_type="application/json")
    if request.method == 'GET':
        data = {
                'name': 'Vitor 1',
                'location': 'Finland',
                'is_active': True,
                'count': 29
        }
        return JsonResponse(data)
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