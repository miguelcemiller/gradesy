from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.

def home_view(request):
    return render(request, 'application/home.html')

def process_essay(request):
    # unpack request from front end:
    input_essay = request.POST['essay']

    # test essay
    print(input_essay)

    # pack response:
    response = {
        'essay' : input_essay,
    }

    return JsonResponse(response)