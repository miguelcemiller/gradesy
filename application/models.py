from django.db import models
import uuid
from django.contrib.auth.models import User

# Create your models here.
class Data(models.Model):
    username = models.CharField(max_length=100, blank=True, null=True)
    plagiarised_words = models.TextField(blank=True, null=True)
    plagiarism_score = models.IntegerField(blank=True, null=True)

    essay = models.TextField(blank=True, null=True)

    created = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(default=uuid.uuid4, unique=True, primary_key=True, editable=False)

    def __str__(self):
        return str(self.username) or ''