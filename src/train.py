import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from src.model import build_srcnn
from src.data_loader import load_div2k_images, load_custom_images

def train_model(dataset_type='div2k', epochs=50, batch_size=16, split_ratio=0.8):
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

        num_samples = lr_images.shape[0]
        split_idx = int(num_samples * split_ratio)
        
        X_train = lr_images[:split_idx]
        y_train = hr_images[:split_idx]
        X_val = lr_images[split_idx:]
        y_val = hr_images[split_idx:]
        
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
    checkpoint = ModelCheckpoint('best_model..keras', monitor='val_loss', save_best_only=True, verbose=1)

    try:
        print("Iniciando el entrenamiento...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
        print("Entrenamiento completado.")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

    print("Entrenamiento completado. El mejor modelo ha sido guardado como 'best_model.keras'")    