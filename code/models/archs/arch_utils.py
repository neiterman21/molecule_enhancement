import torch
from PIL import Image

def save_tensor_as_image(tensor, filename):
    """
    Save a single-channel PyTorch tensor as a grayscale image.

    Args:
    tensor (torch.Tensor): A 2D tensor representing a grayscale image.
    filename (str): The filename for the saved image.
    """
    # Ensure the tensor is on CPU and convert to numpy
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255

    image_array = tensor.cpu().numpy()

    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array.astype('uint8'), mode='L')

    # Save the image
    image.save(filename)