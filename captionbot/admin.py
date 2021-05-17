from django.contrib import admin
from captionbot import models

# Register your models here.

admin.site.register([
    models.UploadImage
])
