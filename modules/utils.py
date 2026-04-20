#PISTA: es esta la mejor forma de hacer una matmul?
import numpy as np

# Aqui se calcula la salida de la capa Dense realizando la multiplicación entre la matriz
# de entrada A y la matriz de pesos B, Después, añade el vector de sesgo bias a cada 
# fila del resultado de forma vectorizada.
def matmul_biasses(A, B, C, bias):
    
    return (A @ B + bias).astype(np.float32)
