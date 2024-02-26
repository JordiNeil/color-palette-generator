
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import json
from deco import synchronized, concurrent
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth


def load_images(root_path, file_names, normalize=True):
    """
    Retorna una lista con arreglos de números. Cada arreglo representa una imagen RGB.
    
    Parametros:
    root_path : str
        Directorio raíz de las imágenes.
    file_names : list
        Lista con los nombres de los archivos de imágenes.
    flatten : bool, opcional
        Si es True, aplana las imágenes a una lista de píxeles de tres dimensiones.
    normalize : bool, opcional
        Si es True, normaliza los valores de píxeles al rango [0, 1].
    """
    image_list = list()
    for file in file_names:
        img = cv2.imread(os.path.join(root_path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if normalize:
            img = img / 255.0
        image_list.append(img)
    return image_list

def flat_image(image):
    """
    Aplana una imagen a una lista de píxeles de tres dimensiones.
    
    Parametros:
    image : numpy.ndarray
        Arreglo de números que representa una imagen RGB.
    """
    return image.reshape((-1, 3))

@concurrent
def train(X, k, model):
    """
    Entrena un modelo de clustering
    
    Parametros:
    X : np.array
        El arreglo con los datos
    k : int
        Número de clusters
    model : str
        "kmeans" o "kmedoids", especifica el modelo a entrenar.
    """
    if model == "kmeans":
        model_k = KMeans(n_clusters=k, max_iter=100, n_init=10, random_state=0)
    # else:
        # model_k = KMedoids(n_clusters=k, max_iter=300, random_state=0)
    # Entrenamos el modelo
    model_k.fit(X)
    score = silhouette_score(X, model_k.labels_)
    return score

@synchronized
def silhouette_plot(X, model, k_min=4, k_max=8):
    """
    Genera la gráfica con el coeficiente de la silueta
    
    Parametros:
    X : np.array
        El arreglo con los datos
    model : str
        "kmeans" o "kmedoids", especifica el modelo a entrenar.
    k_min : int
        Valor mínimo para k
    k_max : int
        Valor máximo para k
    """
    scores = {}
    for k in range(k_min, k_max+1):
        # print('Training model with k: ', k)
        scores[k] = train(X, k, model)
            
    # Graficamos los valores del coeficiente de la silueta
    return scores

def main():
    parser = argparse.ArgumentParser(description='Run parallel Kmeans models')
    parser.add_argument('root_path', type=str, help='Images root path')
    parser.add_argument('kmin', nargs='?', type=str,const=1, default='4', help='Description of param2')
    parser.add_argument('kmax', nargs='?', type=str,const=1, default='8', help='Description of param3')
    parser.add_argument('image_picker', nargs='?', type=str, const=1, default='3', help='The index of the image to select')
    args = parser.parse_args()

    # variable casting
    root_path = args.root_path
    kmin = int(args.kmin)
    kmax = int(args.kmax)
    image_picker = int(args.image_picker)

    print("root_path: ", root_path)
    print("kmin: ", kmin)
    print("kmax: ", kmax)
    print("image_picker: ", image_picker)

    img_files = os.listdir(root_path)
    img_files


    # Cargar imágenes
    print('Loading images')
    img_list = load_images(root_path, img_files, normalize=True)
    # Resize image
    print('Resizing images')
    img_resized = cv2.resize(img_list[image_picker], (100, 100)) # This is fixed

    # img_original_flatten = flat_image(img_list[image_picker])
    # img_original_flatten.shape

    img_resized_flatten = flat_image(img_resized)
    img_resized_flatten.shape
    
    print("Training completed")
    new_scores = silhouette_plot(img_resized_flatten, "kmeans", kmin, kmax)
    serialized_output = json.dumps(new_scores)
    print(serialized_output)

    return serialized_output

if __name__ == "__main__":
    print("Training models in parallel...")
    output = main()

