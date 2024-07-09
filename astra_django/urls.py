from django.contrib import admin
from django.urls import path
from qa_app.views import index, ask_question

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('qa/ask/', ask_question, name='ask_question'),
]
