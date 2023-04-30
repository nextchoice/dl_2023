from django.urls import path
from .views import *

app_name='app'

urlpatterns = [
    path('', index),
    path('catordog/', catordog),
    path('predict/', predict),
    path('whichcatordog/', whichcatordog),
    path('predictwithinception/', predict_with_inception),
]