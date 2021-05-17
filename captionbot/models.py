from django.db import models

# Create your models here.

def nameFile(instance, filename):
    return '/'.join(['captionbot', str(instance.name), filename])


class UploadImage(models.Model):

    name = models.CharField(max_length = 100)
    image = models.ImageField(upload_to = nameFile, blank = True, null = True)
    result = models.CharField(max_length = 256)


    def __str__(self):
        return self.name
