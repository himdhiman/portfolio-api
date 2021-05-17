from django.shortcuts import render
from captionbot import models
from rest_framework.response import Response
import os
from rest_framework.views import APIView
from captionbot import serializers
import json
import shutil

from captionbot.helper import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImageViewSet(APIView):
    queryset = models.UploadImage.objects.all()
    serializer_class = serializers.ImageSerializer

    def post(self, request, *args, **kwargs):
        os.chdir(os.path.join(BASE_DIR, "media"))
        shutil.rmtree('captionbot')
        os.chdir(BASE_DIR)
        file = request.data['image']
        image = models.UploadImage.objects.create(image=file)
        out = predict()
        return Response({'message': out})