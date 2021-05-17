from django.urls import path
from sentiment import views

urlpatterns = [
    path('', views.Index)
] 