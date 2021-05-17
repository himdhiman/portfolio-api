from django.shortcuts import render
from rest_framework.response import Response
import os, shutil
from rest_framework.views import APIView
from covidapi import models, serializers
from covidapi.helper import *


class ImageViewSet(APIView):
    queryset = models.UploadImage.objects.all()
    serializer_class = serializers.ImageSerializer

    def post(self, request, *args, **kwargs):
        os.chdir(os.path.join(BASE_DIR, "media"))
        if(os.path.isdir(os.path.join(BASE_DIR, "media", "covid"))):
            shutil.rmtree('covid')
        os.chdir(BASE_DIR)
        file = request.data['image']
        image = models.UploadImage.objects.create(image=file)
        out = predict()
        return Response({'message': out})




def predict():
    prediction = getResult()
    if(prediction[1] == 1.0):
        return "Covid-19 Positive"
    return "Covid-19 Negative"


