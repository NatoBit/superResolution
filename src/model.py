import tensorflow as tf
from tensorflow.keras import layers, models

def build_srcnn():
    """
    Construye y compila un modelo CNN para super resolución de imágenes.

    La arquitectura se basa en tres capas convolucionales:
    - Primera capa: 64 filtros con tamaño de 9x9 y activación ReLU.
    - Segunda capa: 32 filtros con tamaño de 5x5 y activación ReLU.
    - Tercera capa (salida): 3 filtros con tamaño de 5x5 y activación lineal (salida de 3 canales para imágenes RGB).

    El modelo se compila con el optimizador 'Adam' y la función de pérdida 'mean_squared_error'.

    Returns:
        model (tensorflow.keras.Model): Modelo SRCNN compilado.
    """
    
    # Inicializamos un modelo secuencial
    model = models.Sequential()

    # Capa convolucional 1:
    # - 64 filtros
    # - Tamaño de filtro: 9x9
    # - Función de activación: ReLU (Rectified Linear Unit)
    # - Padding: 'same' (se mantiene el tamaño de la imagen en esta capa)
    # - input_shape: acepta imágenes RGB de cualquier tamaño (None, None, 3)
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 3)))

    # Capa convolucional 2:
    # - 32 filtros
    # - Tamaño de filtro: 5x5
    # - Función de activación: ReLU
    # - Padding: 'same' (mantiene el tamaño de la imagen)
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))

    # Capa de salida:
    # - 3 filtros (ya que queremos obtener una imagen RGB con 3 canales)
    # - Tamaño de filtro: 5x5
    # - Activación: 'linear' (sin activación no lineal, ya que estamos prediciendo valores de píxeles)
    # - Padding: 'same' para asegurar que el tamaño de la salida coincide con la entrada
    model.add(layers.Conv2D(3, (5, 5), activation='linear', padding='same'))

    # Compilación del modelo:
    # - Optimizador: Adam, un optimizador eficiente que ajusta los pesos basándose en la estimación de los primeros y segundos momentos.
    # - Función de pérdida: mean_squared_error, adecuada para problemas de regresión en los que se busca minimizar la diferencia entre los valores de predicción y los reales.
    # - Métricas: 'accuracy', para monitorizar el rendimiento, aunque en la super resolución es común usar métricas como PSNR o SSIM.
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
