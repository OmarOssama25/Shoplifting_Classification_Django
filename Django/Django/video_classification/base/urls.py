from django.urls import path
from .views import process_video

urlpatterns = [
    path('', process_video, name='upload_video'),  # يمكن تعديل هذا إذا كنت ترغب في مسار مختلف
    path('process/', process_video, name='process_video'),
]
