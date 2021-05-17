from django.db import models


# Create your models here.

def nameFile(instance, filename):
    return '/'.join(['covid/imgdata', str(instance.name), filename])


class UploadImage(models.Model):
    lookup = (
        ("NA", "None"),
        ("P", "Positive"),
        ("N", "Negative"),
    )

    name = models.CharField(max_length = 100)
    image = models.ImageField(upload_to = nameFile, blank = True, null = True)
    result = models.CharField(max_length = 10, choices = lookup, default = "NA")


    def __str__(self):
        return self.name
