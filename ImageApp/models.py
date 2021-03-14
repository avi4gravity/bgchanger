from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.
# Model for images upload
class Images(models.Model):
    #user_name = models.ForeignKey(User)
    #user_name = models.OneToOneField(User, on_delete=models.CASCADE)
    user_name = models.ForeignKey('auth.User', on_delete=models.CASCADE,default=1)
    #user_name = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,default=1)
    #user_name = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    def __str__(self):
        return str(f'{self.pk} - {self.user_name}')

class Backgrounds(models.Model):
    img_desc = models.TextField()
    back_img = models.ImageField(default='default.jpg', upload_to='background_pics/')
    img_owner = models.ForeignKey('auth.User', on_delete=models.CASCADE,default=1)
    def __str__(self):
        return f'{self.pk} - {self.img_desc}' 

    @property
    def imageURL(self):
        try:
            url = self.back_img.url
        except:
            url = ''
        return url
