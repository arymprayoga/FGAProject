"""bigdatadigitalent URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from task import views

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^$', views.index),
    path('week1', views.function_week1),
    path('week21', views.function_week2_task1),
    path('week22', views.function_week2_task2),
    path('week3', views.function_week3),
    path('week4', views.function_week4),
    path('week41', views.function_week4_task1),
    path('week5', views.function_week5),
    path('week6', views.function_week6),
    path('week7', views.function_week7),
    path('week8', views.function_week8),
    path('week51', views.function_week5_task1),
    path('week52', views.function_week5_task2),
    path('api&<str:a>&<str:b>&<str:c>', views.api),
]
