# Generated by Django 4.0.3 on 2022-06-05 06:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0003_data_user'),
    ]

    operations = [
        migrations.RenameField(
            model_name='data',
            old_name='user',
            new_name='username',
        ),
    ]
