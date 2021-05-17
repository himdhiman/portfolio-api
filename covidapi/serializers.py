from rest_framework import serializers
from covidapi import models

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.UploadImage
        fields = '__all__'