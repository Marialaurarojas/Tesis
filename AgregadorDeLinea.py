# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:58:00 2024

@author: Maria Laura

Este programa agrega la primera linea a todos los csv dados por el osciloscopio
es necesario agregar esta linea ya que es la que identifica los datos contenidos en el archivo
"""

import csv



# Itera sobre los números entre 0 y 6
for i in range(0, 7):
    # Itera sobre los números 1 y 2, estos numeros son los canales con los que se trabajó
    for j in range(1, 3):
        # Crea el nombre del archivo
        filename = f"F000{i}CH{j}.csv"

        #necesitamos agregar una nueva fila al archivo csv original para identificar los datos de cada columna
        data = ['a', 'b', 'c','x', 'y', 'd']

        # Lee el archivo original y almacena los datos
        with open(filename, 'r') as f:
           original = list(csv.reader(f))

      # Abre el archivo en modo de escritura )
        with open(filename, 'w', newline='') as f:
         writer = csv.writer(f)
         # Escribe los nuevos datos en el archivo
         writer.writerow(data)
          # Escribe los datos originales después de los nuevos datos
         writer.writerows(original)