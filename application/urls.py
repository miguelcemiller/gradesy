from django.urls import path
from django.shortcuts import redirect
from . import views
from django.views.generic import RedirectView

urlpatterns = [
    path('', views.home_view, name='home'),
    path('results', views.results_view, name='results'),
    path('check_essay', views.check_essay, name='check_essay'),
    path('get_data', views.get_data, name='get_data'),
]