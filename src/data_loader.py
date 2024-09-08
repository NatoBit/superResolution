import tensorflow as tf
import tensorflow_datasets as tfds
import os

def preprocess_image(image, scale=4):
    """
    Preprocesa una imagen de alta resolución para simular la baja resolución.
    
    Redimensiona la imagen de alta resolución a una baja resolución utilizando
    el método bicúbico, luego la redimensiona nuevamente a la resolución original.
    
    Args:
        image (tf.Tensor): Imagen de alta resolución en formato tensor.
        scale (int): Factor de escala para la reducción de resolución. Por defecto es 4.
        
    Returns:
        tf.Tensor: Imagen de baja resolución simulada en formato tensor.
    """

    lr_image = tf.image.resize(image, (image.shape[0] // scale, image.shape[1] // scale), method='bicubic')
    lr_image = tf.image.resize(lr_image, (image.shape[0], image.shape[1]), method='bicubic')
    """
    Nota: El método bicúbico es una técnica de interpolación utilizada para redimensionar imágenes. Calcula el valor de un píxel en base a una 
    combinación ponderada de los píxeles vecinos más cercanos, lo que proporciona resultados más suaves y menos pixelados en comparación con 
    métodos más simples como el bilineal.
    """
    return lr_image

def load_div2k_images(scale=4):
    """
    Carga y preprocesa el dataset DIV2K usando TensorFlow Datasets.
    
    Descarga el dataset DIV2K y prepara las imágenes de alta y baja resolución.
    
    Args:
        scale (int): Factor de escala para la reducción de resolución. Por defecto es 4.
        
    Returns:
        tuple: Una tupla de dos tensores; el primero contiene imágenes de baja resolución,
               y el segundo contiene imágenes de alta resolución.
    """

    # Cargar el dataset DIV2K
    dataset, info = tfds.load('div2k/4x', with_info=True, as_supervised=True, split='train+validation', shuffle_files=True, batch_size=-1)
    
    # Obtener las imágenes del dataset
    images = dataset['image']
    # Convertir las imágenes a valores de punto flotante en el rango [0, 1]
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    # Preparar las imágenes de alta y baja resolución
    hr_images = images
    # tf.map_fn: Es una función de TensorFlow que aplica una función dada a cada elemento en un tensor a lo largo de un eje específico.
    lr_images = tf.map_fn(lambda img: preprocess_image(img, scale), hr_images, dtype=tf.float32)

    return lr_images, hr_images

