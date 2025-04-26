from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_survival, name='predict'),
path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('login/', views.login_page, name='login'),
    path('predict/', views.predict_survival, name='predict'),
    path('history/', views.prediction_history, name='history'),
]
