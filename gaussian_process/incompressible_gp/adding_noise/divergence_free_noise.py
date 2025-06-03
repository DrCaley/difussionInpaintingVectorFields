import numpy as np

def curl_of_gradient_noise(shape):
    """Generates divergence-free noise using the curl of a gradient method.
    Args:
        shape (tuple): Shape of the noise field.
    Returns:
        numpy.ndarray: Divergence-free noise field.
    """
    if len(shape) == 2:  # 2D case
        scalar_field_1 = np.random.rand(*shape)
        scalar_field_2 = np.random.rand(*shape)

        grad_1_x, grad_1_y = np.gradient(scalar_field_1)
        grad_2_x, grad_2_y = np.gradient(scalar_field_2)

        # Calculate curl (2D curl is a scalar)
        curl = grad_2_x - grad_1_y

        return curl

    elif len(shape) == 3:  # 3D case
        scalar_field_1 = np.random.rand(*shape)
        scalar_field_2 = np.random.rand(*shape)
        scalar_field_3 = np.random.rand(*shape)

        grad_1_x, grad_1_y, grad_1_z = np.gradient(scalar_field_1)
        grad_2_x, grad_2_y, grad_2_z = np.gradient(scalar_field_2)
        grad_3_x, grad_3_y, grad_3_z = np.gradient(scalar_field_3)

        # Calculate curl
        curl_x = grad_2_z - grad_3_y
        curl_y = grad_3_x - grad_1_z
        curl_z = grad_1_y - grad_2_x

        return np.stack([curl_x, curl_y, curl_z])

    else:
         raise ValueError("Shape must be 2D or 3D")
