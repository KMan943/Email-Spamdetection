import pickle
from django.shortcuts import render

path_model = "static/predictors/NaiveBayes_model.pkl"
path_vectorizer = "static/predictors/vectorizer.pkl"

with open(path_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_model, 'rb') as f:
    model = pickle.load(f)

# Create your views here.
def predict(request):
    return render(request , 'base/predict.html')

def results(request):
    if request.method == "GET":
        Email_text = request.GET.get('Email')
        # print(Email_text)
        text = [Email_text]
        X = vectorizer.transform(text)
        ans = model.predict(X)

        context = {'ans' : ans , 'Email' : Email_text}
    else:
        context = {'ans' : -1 , 'Email' : Email_text}

    return render(request , 'base/results.html' , context=context)
