import numpy as np
cimport numpy as np

# --- INICIO BLOQUE GENERADO CON IA --- #

# --- Este bloque de codigo hace la misma funcion como la de maxpool2d.py, que tambien recorre por cada ventana de pooling de la entrada 
# --- y calcula su valor máximo, los bucles esta implementado en cython para que ejecuta de una forma mas eficiente como la de C, en vez de python.

def maxpool_forward_cython(np.ndarray[np.float32_t, ndim=4] input,
                           int kernel_size,
                           int stride):
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]

    cdef int KH = kernel_size
    cdef int KW = kernel_size
    cdef int SH = stride
    cdef int SW = stride

    cdef int out_h = (H - KH) // SH + 1
    cdef int out_w = (W - KW) // SW + 1

    cdef np.ndarray[np.float32_t, ndim=4] output = np.zeros((B, C, out_h, out_w), dtype=np.float32)

    cdef int b, c, i, j, h_start, w_start
    cdef int ii, jj
    cdef float max_val, val

    for b in range(B):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * SH
                    w_start = j * SW

                    max_val = input[b, c, h_start, w_start]

                    for ii in range(KH):
                        for jj in range(KW):
                            val = input[b, c, h_start + ii, w_start + jj]
                            if val > max_val:
                                max_val = val

                    output[b, c, i, j] = max_val

    return output

# --- FIN BLOQUE GENERADO CON IA --- #