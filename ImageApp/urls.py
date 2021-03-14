from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    #path('upload',views.UploadView,name='upload'),
    path('images',views.ImagesView,name='images'),
    path('upload',views.file_upload_view,name='upload'),
    path('ch_bg',views.change_bg_view,name='change_bg'),
    path('upload1',views.two_bg_change,name='change_bg2'),
    #path('backgrounds/<int:imgID>/',views.background,name='backgrounds'),
    path('imagesselected/<int:user_img>/<int:back_img>/',views.ImagesSelected,name='images_selected'),
]