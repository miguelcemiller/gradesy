from django.urls import path
from django.shortcuts import redirect
from . import views
from django.views.generic import RedirectView

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='home')),
    path('home', views.home_view, name='home'),
    path('essay', views.process_essay, name='process_essay'),
]