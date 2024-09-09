import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_srcnn
from data_loader import load_div2k_images, load_custom_images

def train_model(dataset_type='div2k', epochs=50, batch_size=16):
    """
    Entrena el modelo de super-resolución basado en CNN.

    Args:
        dataset_type (str): Tipo de conjunto de datos ('div2k' o 'custom').
        epochs (int): Número de épocas para entrenar el modelo.
        batch_size (int): Tamaño del lote para el entrenamiento.
    """
    
    if dataset_type == 'div2k':
        # Cargar imágenes de entrenamiento y validación usando la función de DIV2K
        lr_images, hr_images = load_div2k_images()

        # Aquí no necesitamos dividir los datos, ya que el split 'train+validation' está predefinido
        # Podemos usar un subset del dataset cargado directamente

        # Por ejemplo, si 'train' y 'validation' ya están en el split, podrías hacer algo como:
        X_train = lr_images['train']
        y_train = hr_images['train']
        X_val = lr_images['validation']
        y_val = hr_images['validation']
        
    else:
        # Cargar imágenes personalizadas
        data_path = 'data/custom/'  # Establecer ruta de datos personalizada
        images = load_custom_images(data_path)
        lr_images = np.array([img[0] for img in images])
        hr_images = np.array([img[1] for img in images])

        # Normalizar las imágenes personalizadas
        lr_images = lr_images / 255.0
        hr_images = hr_images / 255.0

        # Dividir las imágenes personalizadas en entrenamiento y validación
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(lr_images, hr_images, test_size=0.2, random_state=42)

    # Construir el modelo SRCNN
    model = build_srcnn()

    # Guardar el mejor modelo durante el entrenamiento
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Entrenar el modelo
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    print("Entrenamiento completado. El mejor modelo ha sido guardado como 'best_model.h5'")    