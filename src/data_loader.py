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

def load_div2k_images(scale=4, target_size=(360, 640)):
    """
    Carga y preprocesa el dataset DIV2K usando TensorFlow Datasets.
    
    Descarga el dataset DIV2K y prepara las imágenes de alta y baja resolución.
    
    Args:
        scale (int): Factor de escala para la reducción de resolución. Por defecto es 4.
        
    Returns:
        tuple: Una tupla de dos tensores; el primero contiene imágenes de baja resolución,
               y el segundo contiene imágenes de alta resolución.
    """

    # Cargar el dataset DIV2K con una configuración válida, como 'bicubic_x4'
    dataset, info = tfds.load('div2k/bicubic_x4', with_info=True, as_supervised=True, split='train+validation', shuffle_files=True)
    
    # Desempaquetar las imágenes de baja resolución (LR) y alta resolución (HR)
    lr_images, hr_images = [], []
    
    for lr_image, hr_image in dataset:
        # Redimensionar las imágenes al tamaño objetivo
        lr_image = tf.image.resize(lr_image, target_size, method='bicubic')
        hr_image = tf.image.resize(hr_image, target_size, method='bicubic')

        # Convertir las imágenes a valores de punto flotante en el rango [0, 1]
        lr_images.append(tf.image.convert_image_dtype(lr_image, dtype=tf.float32))
        hr_images.append(tf.image.convert_image_dtype(hr_image, dtype=tf.float32))

    # Convertir las listas a tensores
    lr_images = tf.stack(lr_images)
    hr_images = tf.stack(hr_images)
    
    return lr_images, hr_images

def load_custom_images(data_path, scale=4):
    """
    Carga y preprocesa imágenes desde un directorio personalizado.
    
    Lee las imágenes en formato PNG desde el directorio especificado y las preprocesa
    para crear imágenes de baja resolución simuladas.
    
    Args:
        data_path (str): Ruta al directorio que contiene las imágenes personalizadas.
        scale (int): Factor de escala para la reducción de resolución. Por defecto es 4.
        
    Returns:
        tuple: Una tupla de dos tensores; el primero contiene imágenes de baja resolución,
               y el segundo contiene imágenes de alta resolución.
    """
    
    # Obtener las rutas de las imágenes en el directorio personalizado
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]
    # Leer y decodificar las imágenes
    images = [tf.image.decode_image(tf.io.read_file(img_path)) for img_path in image_paths]
    # Convertir la lista de imágenes en un tensor
    images = tf.stack(images)

    # Preparar las imágenes de alta y baja resolución
    hr_images = images
    lr_images = tf.map_fn(lambda img: preprocess_image(img, scale), hr_images, dtype=tf.float32)

    return lr_images, hr_images