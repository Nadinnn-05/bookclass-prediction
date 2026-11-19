from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("penjelasan/", views.penjelasan, name="penjelasan"),
    path("penjelasan-prediksi/", views.penjelasan_prediksi_view, name="penjelasan_prediksi"),
    path("status-ai/", views.status_ai, name="status_ai"),
]