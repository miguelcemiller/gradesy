# Generated by Django 4.0.3 on 2022-06-07 05:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0013_data_lexical_complexity_words'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='content_words',
            field=models.TextField(blank=True, null=True),
        ),
    ]
