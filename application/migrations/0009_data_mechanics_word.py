# Generated by Django 4.0.3 on 2022-06-06 19:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0008_data_grammar_matches'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='mechanics_word',
            field=models.TextField(blank=True, null=True),
        ),
    ]