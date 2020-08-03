from django.contrib import admin
from django.urls import path
from mysite.core import views
from . import settings
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('', views.home, name = 'home'),
    path('result/', views.result, name = 'result'),
    path('randomise/', views.randomise, name='randomise'),
    path('local/', views.local, name = 'local'), 
    path('admin/', admin.site.urls),
]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.OUTPUT_URL, document_root=settings.OUTPUT_ROOT)

