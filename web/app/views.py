from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

from keras.models import load_model, save_model
from keras.applications import InceptionV3
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create your views here.

def index(request):
    return render(request,'index.html')

def whichcatordog(request):
    return render(request,'whichcatordog.html')

@csrf_exempt
def catordog(request):
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
    else:
        myfile = {'name':'몰라'}

    which = predict(myfile)

    data = {
        "filename": myfile.name,
        "which": which
    }
    return JsonResponse(data)

@csrf_exempt
def predict(request):
    # myfile = request.FILES.get('myfile')
    if request.method == 'POST' and request.FILES['files']:
        myfile = request.FILES['files']
    else:
        myfile = {'name':'몰라'}

    which = predict_image(myfile)

    data = {
        "filename": myfile.name,
        "which": which['result'],
        "proba": which['result_proba']
    }
    return JsonResponse(data)

@csrf_exempt
def predict_with_inception(request):
    print(request.FILES)
    # myfile = request.FILES.get('myfile')
   
    if request.method == 'POST' and request.FILES['files']:
        myfile = request.FILES['files']
    else:
        myfile = {'name':'몰라'}

    which = predict_image_with_inception(myfile)

    data = {
        "filename": myfile.name,
        "which": which['result'],
        "proba": which['result_proba']
    }
    return JsonResponse(data)

def predict_image(img):
    model = load_model('/home/nextchoice/git-dev/dl_2023/dog_or_cat.h6')
    t = plt.imread(img)
    im_rgb = cv.cvtColor(t, cv.COLOR_BGR2RGB)
    im_rgb = cv.resize(im_rgb, (150,150))
    im_rgb = im_rgb.reshape(1,150,150,3)
    predicted_result = model.predict(im_rgb)

    result_proba = np.around(predicted_result[0][0]*100, decimals=2)
    result = np.where(predicted_result > 0.5, 'dog', 'cat')[0][0]
    if result == 'cat':
        result_proba = 100 - np.around(predicted_result[0][0]*100, decimals=2)

    # return np.array2string(np.where(model.predict(im_rgb) > 0.5, 'dog', 'cat'))
    return {'result':result, 'result_proba':result_proba}

def predict_image_with_inception(img):
    base = InceptionV3(
        include_top=False,
        input_shape=(150, 150, 3)
    )

    model = load_model('/home/nextchoice/git-dev/dl_2023/cat_or_dog_with_inception.h5')
    t = plt.imread(img)
    im_rgb = cv.cvtColor(t, cv.COLOR_BGR2RGB)
    im_rgb = cv.resize(im_rgb, (150,150))
    im_rgb = im_rgb.reshape(1,150,150,3)
    dt = base.predict(im_rgb)

    predicted_result = model.predict(dt)

    print(predicted_result)
    result_proba = np.around(predicted_result[0][0] * 100, decimals=2)
    result = np.where(predicted_result > 0.5, 'dog', 'cat')[0][0]
    if result == 'cat':
        result_proba = 100 - np.around(predicted_result[0][0]*100, decimals=2)

    # return np.array2string(np.where(model.predict(im_rgb) > 0.5, 'dog', 'cat'))
    return {'result':result, 'result_proba':result_proba}