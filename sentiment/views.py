from django.shortcuts import render
from django.http import JsonResponse, request
import paralleldots, json
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

paralleldots.set_api_key("Ql3xUPHD2C2XyQCRsBkPWaq65WFcBRShQ48gw9fngWE")

@api_view(['POST'])
def Index(request):
    data = json.loads(request.body.decode('utf-8'))
    sent = data["sentiment"]
    result = paralleldots.sentiment(sent, lang_code='en')['sentiment']
    result = dict(result)
    pos = result['positive']
    neg = result['negative']
    neu = result['neutral']
    data = {
        'positive' : pos,
        'negative' : neg,
        'neutral' : neu
    }
    return JsonResponse(data, safe = False)
