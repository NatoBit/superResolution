import matplotlib.pyplot as plt

def visualize_results(low_res_image, high_res_image, predicted_image):
    plt.figure(figsize=(15, 5))
    
    # Imagen de baja resolución
    plt.subplot(1, 3, 1)
    plt.imshow(low_res_image)
    plt.title('Baja Resolución')
    
    # Imagen de alta resolución (real)
    plt.subplot(1, 3, 2)
    plt.imshow(high_res_image)
    plt.title('Alta Resolución (real)')
    
    # Imagen generada por el modelo
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_image)
    plt.title('Alta Resolución (predicha)')
    
    plt.show()