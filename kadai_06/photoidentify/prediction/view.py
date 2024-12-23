from .forms import ImageUploadForm
from django.shortcuts import render
from django.conf import settings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
from io import BytesIO

def preprocess_image(img_path):
    unknown_img = load_img(img_path, target_size=(224, 224))
    unknown_array = img_to_array(unknown_img)
    unknown_array = preprocess_input(unknown_array)
    unknown_array = np.expand_dims(unknown_array, axis=0) 
    return unknown_array

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img_array = preprocess_image(img_path=img_file)

            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            result = model.predict(img_array)
            prediction = decode_predictions(result)
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})