import argparse
from src.train import train_model

def main():
    """
    Función principal que gestiona el flujo del entrenamiento del modelo de super-resolución.

    - Lee los argumentos de la línea de comandos.
    - Llama a la función `train_model` con los argumentos proporcionados.
    """
    
    # Configuración de argumentos para la línea de comandos
    parser = argparse.ArgumentParser(description='Entrena un modelo de super resolución con DIV2K o un set personalizado.')

    # Añadir argumentos de entrada para el tipo de dataset, número de épocas y tamaño del batch
    parser.add_argument('--dataset_type', type=str, default='div2k', choices=['div2k', 'custom'],
                        help="Tipo de dataset para entrenar ('div2k' o 'custom').")
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas para entrenar.')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del batch para el entrenamiento.')

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamada a la función de entrenamiento con los parámetros proporcionados
    train_model(dataset_type=args.dataset_type, epochs=args.epochs, batch_size=args.batch_size) 

if __name__ == '__main__':
    # Este bloque asegura que la función `main()` se ejecute cuando se llame a este script directamente
    main()