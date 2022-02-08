#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:26:30 2021

@author: AnahiRomo

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from osgeo import gdal
from sklearn.cluster import KMeans

def busca_archivos(directorio, cadena, cadena2 = None):
    '''
    Busca archivos que tengan la 'cadena' en el nombre dentro del 'directorio'
    y sus subdirectorios y retorna una lista con sus nombres.
    Recibe dos parámetros: nombre del directorio y cadena
    Retorna una lista con los nombres de los archivos buscados
    '''
    lista  = []

    for name, dirs, files in os.walk(directorio):
        for name in files:
            if cadena2:  # dejo posibilidad de poner otra cadena x + precisión
                if (cadena in name) and (cadena2 in name):
                    lista.append(name)
            else:
                if cadena in name:
                    lista.append(name)
        for name in dirs:
            if cadena2:
                if (cadena in name) and (cadena2 in name):
                    lista.append(name)
            else:
                if cadena in name:
                    lista.append(name)

    return lista

def cargar_banda(carpeta, banda, zona, extra = 0):
    ''' Busca y carga una banda pedida de imagen raster desde una carpeta dada.
    Parámetros: 'carpeta', 'banda', opcional: 'extra', string; 'zona' es una
    4-tupla con los valores de lat-long de zona a recortar:
    zona =(xmin, xmax, ymin, ymax)
    '''
    if banda:   # elijo el archivo que corresponde a la banda
        band = int(banda)
        num_banda = 'band' + str(band)
        nombre_banda = num_banda + '.tif'
    # busco la imagen raster de la banda elegida en la carpeta donde las puse
    if extra:
        lista = busca_archivos(carpeta, num_banda, extra)
        r = str(extra) +'_b' + str(band) + '.tif'
    else:
        lista = busca_archivos(carpeta, num_banda)

    # sé que hay un solo archivo en esa carpeta, y es lista[0]
    file = lista[0]

    # cargo al raster original y lo convierto a coordenadas lat long
    # nombre_banda es el archivo de imagen, una vez 'traducido' a lat/long
    f1 = gdal.Warp(nombre_banda, file, dstSRS = 'EPSG:4326')

    # recorto la imagen sólo a la zona de interés especificada en la tupla zona
    # 'r' es el nombre del raster recortado
    f2 = gdal.Warp(r, f1, outputBounds = (float(zona[0]), float(zona[2]),
                                          float(zona[1]), float(zona[3])))
    # convierto archivo imagen recortada a np array
    archivo = f2.ReadAsArray()
    # borro archivos que no necesito
    os.remove(str(nombre_banda))
    os.remove(str(r))
    del f1, f2

    return archivo   # 'archivo' es un array de numpy

def ndvi(carpeta, banda_rojo, banda_nir, zona, satelite):
    '''
    Calcula el índice NDVI para las imágenes guardadas en la carpeta usando la
    banda roja y la infrarroja, según la definición.
    Devuelve el np.array resultante
    Parámetros: carpeta, nro de banda Rojo y nro de banda IR, año si corresponde
    '''

    b_red = cargar_banda(carpeta, banda_rojo, zona, satelite)  # banda 4 (ROJO)
    b_nir = cargar_banda(carpeta, banda_nir, zona, satelite)   # banda 5 (NIR)

    # defino el array ndvi, pero si b_red + b_ir = 0. -> ndvi = -999.
    ndvi = np.where((b_nir + b_red) == 0., -999., (b_nir - b_red) /
                    (b_nir + b_red))

    return ndvi     # 'ndvi' es un array de numpy

def snow_i(carpeta, banda_rojo, banda_swir1, zona, fecha):
    ''' Calcula el snow index para las bandas contenidas en la carpeta',
    según su definición. Devuelve el np.array resultante.
    Parámetros: carpeta, banda roja y banda swir1'''
    b_red = cargar_banda(carpeta, banda_rojo, zona, fecha)   # cargo banda rojo
    b_swir1 = cargar_banda(carpeta, banda_swir1, zona, fecha) #cargo band swir1

    # defino array snow_i, si banda swir1 = 0 -> si = -999.
    snow_i = np.where(b_swir1 == 0., -999., b_red / b_swir1)

    return snow_i     # 'snow_i' es un array de numpy

def nbr(carpeta, banda_nir, banda_swir2, zona, fecha):
    '''Calcula el Normalized Burn Ratio para las bandas nir y swir2 contenidas
    en la 'carpeta', devuelve un np.array con los valores.
    Parámetros: carpeta, bandas nir y swir2'''
    b_nir = cargar_banda(carpeta, banda_nir, zona, fecha)     #cargo banda nir
    b_swir2 = cargar_banda(carpeta, banda_swir2, zona, fecha) #cargo band swir2

    # defino nbr, si el denominador se anula, asigno -999. al nbr
    nbr = np.where((b_nir + b_swir2) == 0., -999., (b_nir - b_swir2) /
                   (b_nir + b_swir2))

    return nbr      # 'nbr' es un array de numpy

def defo_noa(carpeta, banda_red, banda_nir, zona_noa):
    '''Recortar las imágenes a la zona comprendida entre las coordenadas
    24.867°N, 64.539°W, 25.290°S, 63.829°E
    Calcular el NDVI para cada imagen por año y apilarlas
    Clasificar el apilado de NDVI utilizando el método de k-means con 3 clases
    espectrales
    Reclasificar la escena según:
        * zonas con alto NDVI que no cambiaron (color  RGB: 45, 207, 96)
        * zonas con bajo NDVI que no cambiaron (color  RGB: 255, 255, 191)
        * zonas con disminución de NDVI (color  RGB: 252, 141, 89)
        Imprimir la imagen clasificada obtenida '''

    # calculo ndvi para 1986 y 2017 en la zona recortada
    # ambos son np array
    ndvi_1986 = ndvi(carpeta, banda_red, banda_nir, zona_noa, '19860107')
    ndvi_2017_A = ndvi(carpeta, banda_red, banda_nir, zona_noa, '20171230')
    # recorto para que ambos array tengan la misma cantidad de filas/ columnas
    ndvi_2017 = ndvi_2017_A[:1483, :2490]
    '''
    plt.imshow(ndvi_1986, vmin = np.percentile(ndvi_1986.flatten(),5), vmax
               = np.percentile(ndvi_1986.flatten(), 95), cmap = 'viridis')
    plt.imshow(ndvi_2017, vmin = np.percentile(ndvi_2017.flatten(),5), vmax
              = np.percentile(ndvi_2017.flatten(), 95), cmap = 'inferno')
    '''
    # apilo ambos array
    df_ndvi = pd.DataFrame({'ndvi_1986': ndvi_1986.flatten(),
                            'ndvi_2017': ndvi_2017.flatten()})

    # invoco un clasificador kmeans c/ 3 clusters, fijo semilla x
    kmeans = KMeans(n_clusters = 3, random_state = 7)       # reproducibilidad
    kmeans.fit(df_ndvi)                         # fitea clasificación datos
    ndvi_clasificado = kmeans.predict(df_ndvi)  # entrega los datos clasif

    # paso los colores RGB como tupla normalizada
    color_ndvi = colors.ListedColormap([(252/256, 141/256, 89/256),
                                        (45/256, 207/256, 96/256),
                                        (255/256, 255/256, 191/256)])

    # con los 'limites' apareo los colores con las clases y creo norma de color
    limites = [0, 1, 2, 3]
    norm_ndvi = colors.BoundaryNorm(limites, color_ndvi.N)
    # texto para la figura
    text = ['Cambio NDVI', 'Alto NDVI s/c',  'Bajo NDVI s/c']
    # apareo texto con color
    box = [mpatches.Patch(color = color_ndvi(i), label="{:s}".format(text[i]))
           for i in range(len(text))]
    leyenda = box[::-1]

    # creo gráfico y lo guardo como archivo, además de mostrarlo
    plt.imshow(ndvi_clasificado.reshape(ndvi_1986.shape), cmap = color_ndvi,
               norm = norm_ndvi)
    plt.title(f'Deforestación en NOA')
    plt.legend(handles = leyenda, fontsize = 6.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
    #plt.colorbar(shrink = 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('deforestacion.png', dpi = 300) # guardo figura en archivo
    plt.show()

    return

def glaciar_cuyo(carpeta, banda_rojo, banda_swir1, zona):
    ''' Recortar las imágenes a la zona comprendida entre las coordenadas:
        30.052°N, 70.112°W, 30.610°S, 69.026°E
        Calcular el Índice de Nieve (SI = red/swir1) para cada escena de
        verano / invierno
        Buscar (por inspección visual) un valor UMBRAL para la presencia de
        nieve en verano e invierno y construir un mapa de presencia de nieve
        como (si_invierno > UMBRAL_inv)*2 + (si_verano > UMBRAL_ver)*1
        Imprimir el mapa de nieve con la siguiente clasificación:
        * Zonas sin nieve (color RGB: 239, 243, 255)
        * Zonas con nieve sólo en verano (color RGB: 189, 215, 231)
        * Zonas con nieve sólo en invierno (color RGB: 107, 174, 214)
        * Zonas con nieve todo el año (color RGB: 33, 113, 181) '''

    # calculo SI para ambos escenarios INVIERNO / VERANO en zona recortada
    # ambos son np array
    # busca con una segunda cadena que identifica
    si_ver = snow_i(carpeta, banda_rojo, banda_swir1, zona, '20180104')
    si_inv = snow_i(carpeta, banda_rojo, banda_swir1, zona, '20180715')
    '''
    print(si_ver[500:600,500:600])
    print(si_inv[500:600,500:600])
    print(si_ver[1500:1600,2500:2600])
    print(si_inv[1500:1600,2500:2600])
    '''
    # observando imágenes determino UMBRAL_inv, UMBRAL_ver p/ presencia nieve
    # para ambas estaciones da un valor umbral de 1 = promedio_ver
    si_ver_umbral = si_ver.copy()
    si_inv_umbral = si_inv.copy()

    promedio_ver = np.mean(si_ver)
    promedio_inv = np.mean(si_inv)

    umbral_inv = round(promedio_ver)
    umbral_ver = round(promedio_ver)

    #  reclasifico 'a mano', tipo 'máscara binaria' arriba o abajo del umbral
    si_ver_umbral[si_ver_umbral >= umbral_ver] = 1.
    si_ver_umbral[si_ver_umbral < umbral_ver] = 0.
    si_inv_umbral[si_inv_umbral >= umbral_inv] = 1.
    si_inv_umbral[si_inv_umbral < umbral_inv] = 0.

    # este es la combinación buscada, va de 0.(verano e inv debajo del umbral)
    # a 3.(verano e invierno x encima del umbral), o sea, 4 clases
    si =  (si_ver_umbral * 1. + si_inv_umbral * 2.)

    # apilo ambas bandas en un df y clasifico con un k-means de 4 cluster
    df_si = pd.DataFrame({'si_ver':si_ver.flatten(),'si_inv':si_inv.flatten()})
    kmeans = KMeans(n_clusters = 4, random_state = 7)       # reproducibilidad
    kmeans.fit(df_si)                         # fitea clasificación datos
    si_clasificado = kmeans.predict(df_si)

    # paso los colores RGB como tupla normalizada
    color_si = colors.ListedColormap([(239/256, 243/256, 255/256),
                                       (189/256, 215/256, 231/256),
                                       (107/256, 174/256, 214/256),
                                       (33/256, 113/256, 181/256)])

    # con los 'limites' apareo los colores con las clases y creo norma de color
    limites = [0, 1, 2, 3, 4]
    norm_si = colors.BoundaryNorm(limites, color_si.N)
    # texto para la figura
    texto = ['Zona sin nieve', 'Nieve sólo en verano',
            'Nieve sólo en invierno', 'Nieve todo el año']
    # apareo texto con color
    box = [mpatches.Patch(color = color_si(i), label="{:s}".format(texto[i]))
           for i in range(len(texto))]
    # invierto la lista para que concuerde con el orden del colorbar
    leyenda = box[::-1]

    # creo gráfico y lo guardo como archivo, además de mostrarlo
    plt.imshow(si, cmap = color_si, norm = norm_si)
    plt.title(f'Detección de Glaciares en Cuyo')
    plt.legend(handles = leyenda, fontsize = 6.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('glaciar.png', dpi = 300)
    plt.show()

    plt.imshow(si_clasificado.reshape(si_ver.shape), cmap = color_si,
               norm = norm_si)
    plt.title('Glaciares en Cuyo - Clasificación k-means')
    plt.legend(handles = leyenda, fontsize = 6.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('glaciar_k.png', dpi = 300)
    plt.show()
    return

def incendio_patagonia(carpeta, banda_nir, banda_swir2, zona):
    '''
    Recortar las imágenes a la zona comprendida entre las coordenadas:
    42.054°N, 72.323°W, 42.578°S, 71.347°E
    Calcular un índice de área quemada NBR (NBR = (nir − swir2)/(nir + swir2))
    para cada escena (pre / pos incendio) y nombrarlas como nbr_pre y nbr_post
    Calcular la variación de NBR como  ∆NBR = NBRpre − NBRpost
    Construir e imprimir un mapa según la siguiente clasificación:
    * Recrecimiento alto:  menos de -0.25 (color RGB: 26, 152, 80)
    * Recrecimiento bajo:  de -0,25 a  -0.1 (color RGB: 145, 207, 96)
    * No incendiado: de -0.1 a 0.1 (color RGB: 217, 239, 139)
    * Incendio de baja severidad: de 0.1 a 0.27 (color RGB: 255, 255, 191)
    * Incendio severidad moderada baja: 0.27 a 0.44 (color RGB: 254, 224, 139)
    * Incendio severidad moderada alta: 0.44 a 0.66 (color RGB: 252, 141, 89)
    * Incendio de severidad alta: más de 0.66 (color RGB: 215, 48, 39)
    '''

    # calculo los nbr para la zona recortada y las fechas pedidas
    # ambos son np array
    nbr_pre = nbr(carpeta, banda_nir, banda_swir2, zona, '20150121')
    nbr_pos = nbr(carpeta, banda_nir, banda_swir2, zona, '20150411')
    # calculo la variación de nbr antes y después de los incendios
    delta_nbr = nbr_pre - nbr_pos

    # clasifico los intervalos según lista
    clases_nbr = delta_nbr.copy()  # copio para no sobreescribir
    clases_nbr[clases_nbr >= 0.66] = 6
    clases_nbr[(clases_nbr >= 0.44) & (clases_nbr < 0.66)] = 5
    clases_nbr[(clases_nbr >= 0.27) & (clases_nbr < 0.44)] = 4
    clases_nbr[(clases_nbr >= 0.10) & (clases_nbr < 0.27)] = 3
    clases_nbr[(clases_nbr >= -0.10) & (clases_nbr < 0.10)] = 2
    clases_nbr[(clases_nbr >= -0.25) & (clases_nbr < -0.10)] = 1
    clases_nbr[clases_nbr < -0.25] = 0

    # Creo mapa de colores según lista de clasificación
    # cada color con su código RGB en tupla normalizada
    color_nbr = colors.ListedColormap([(26/256, 152/256, 80/256),
                                       (145/256, 207/256, 96/256),
                                       (217/256, 239/256, 139/256),
                                       (255/256, 255/256, 191/256),
                                       (254/256, 224/256, 139/256),
                                       (252/256, 141/256, 89/256),
                                       (215/256, 48/256, 39/256)])
    # Defino los limites de cada color
    limites = [0, 1, 2, 3, 4, 5, 6, 7]
    norm_nbr = colors.BoundaryNorm(limites, color_nbr.N)

    # genero leyenda para el gráfico
    textos = ['Recrec. alto', 'Recrec. bajo', 'No incendiado',
              'Incendio bajo', 'Incendio medio/bajo',
              'Incendio medio/alto', 'Incendio alto']
    le = [mpatches.Patch(color = color_nbr(i), label="{:s}".format(textos[i]))
           for i in range(len(textos))]
    # invierto orden de los textos para que coincida c/ la escala del colorbar
    leyenda = le[::-1]

    # Genero el grafico con los colores nuevos y con leyenda
    plt.imshow(clases_nbr, cmap = color_nbr, norm = norm_nbr)
    plt.title(f'Incendios en Lago Cholila, Patagonia')
    #plt.legend(handles = leyenda, fontsize = 6.5, loc='upper center',
    #          bbox_to_anchor=(1.15, 0.8), shadow=True)
    plt.legend(handles = leyenda, fontsize = 6.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    #plt.colorbar(shrink = 0.5, orientation = 'horizontal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'incendio.png', dpi = 300)    #guarda gráfico
    plt.show()

    return

def fn_ppal():
    '''
    EL ANILLO ÚNICO PARA GOBERNARLOS A TODOS...  :)
    '''

    # las imágenes están en 'TRABAJO FINAL' q es el working directory
    carpeta = '..'
    zona_noa = (-64.539, -63.829, -25.290, -24.867)
    zona_cuyo = (-70.112, -69.026, -30.610, -30.052)
    zona_patag = (-72.323, -71.347, -42.578, -42.054)
    banda_red = 4
    banda_nir =  5
    banda_swir1 = 6
    banda_swir2 = 7
    defo_noa(carpeta, banda_red, banda_nir, zona_noa)
    glaciar_cuyo(carpeta, banda_red, banda_swir1, zona_cuyo)
    incendio_patagonia(carpeta, banda_nir, banda_swir2, zona_patag)

    return

if __name__ == '__main__':

    fn_ppal()

    '''
    RECORDAR LO SGTE!!!
    return _gdal.wrapper_GDALWarpDestName (* args)
    SystemError: <función incorporada wrapper_GDALWarpDestName> devolvió NULL
    sin establecer un error
    Si la imagen de entrada y el código están en el mismo directorio, el
    problema está resuelto. Alternativamente, puede usar la ruta al archivo de
    entrada, pero debe ser la ruta desde la raíz a la imagen, no el directorio
    donde se ejecuta el código.'''