from django.urls import path
from captionbot import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.ImageViewSet.as_view())
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)