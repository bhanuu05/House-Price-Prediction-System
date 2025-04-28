from django.shortcuts import render;
import pickle
import os


def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')


model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'house_price_model.pkl')
model = pickle.load(open(model_path, 'rb'))

def result(request):
    if request.method == 'GET':
        n1 = float(request.GET['n1'])
        n2 = float(request.GET['n2'])
        n3 = float(request.GET['n3'])
        n4 = float(request.GET['n4'])
        n5 = float(request.GET['n5'])

        pred = model.predict([[n1, n2, n3, n4, n5]])
        return render(request, 'result.html', {'result2': pred[0]})


