from django.shortcuts import render
from django.http import request, JsonResponse
from musicgen.helper import *
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def run(request):
    if(request.method == "POST"):
        predict()
        return JsonResponse({"Status" : "Success"})
    return JsonResponse({"Status" : "Failed"})