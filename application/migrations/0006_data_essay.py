# Generated by Django 4.0.3 on 2022-06-05 07:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0005_alter_data_username'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='essay',
            field=models.TextField(blank=True, null=True),
        ),
    ]
