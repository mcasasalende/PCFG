from django.db import models
from django import forms

# Create your models here.
class MyModelName(models.Model):
    """
    Una clase típica definiendo un modelo, derivado desde la clase Model.
    """

    # Campos
    my_field_name = models.CharField(max_length=20, help_text="Enter field documentation")
    

    # Metadata
    class Meta:
        ordering = ["-my_field_name"]

    # Métodos
    def get_absolute_url(self):
         """
         Devuelve la url para acceder a una instancia particular de MyModelName.
         """
         return reverse('model-detail-view', args=[str(self.id)])

    def __str__(self):
        """
        Cadena para representar el objeto MyModelName (en el sitio de Admin, etc.)
        """
        return self.field_name



class Rule(models.Model):
    LHS = models.CharField(max_length=100, default='S')
    RHS = models.CharField(max_length=100, default = 'null')
    totalPret = models.IntegerField(default = 0)
    totalTer = models.IntegerField(default = 0)
    prob = models.DecimalField(max_digits=10, decimal_places=2, default = 0.0)
    modelo = models.CharField(max_length=100, default = 'null')

class TrainModel(models.Model):
    model = models.CharField(max_length=100, default = 'null')
    numPasswords = models.IntegerField(default = 0)
    numRules = models.IntegerField(default = 0)