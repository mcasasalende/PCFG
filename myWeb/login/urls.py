from django.conf import urls
from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('passwd/', views.passwd),
    path('rules/', views.password_rules, name='Rules'),
    path('show/', views.show_rules),
    path('analyze/', views.analize_passwd),
    path('analyze/compare', views.compare, name='compare'),
    path('loading/', views.loading_screen, name='loading_screen'),
    #path('', views.upload_file, name='upload')
    path('upload/', views.upload_file, name='upload')
]