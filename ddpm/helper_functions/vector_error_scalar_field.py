import torch

def compute_error_field(vector_field_real_x, vector_field_real_y, vector_field_predicted_x, vector_field_predicted_y):
    """
    Computes the error vector field between actual and predicted 2D vector fields,
    scaled by the magnitude of the actual data vectors.

    Inputs:
        vector_field_real_x: Tensor of shape (H, W), real x-component
        vector_field_real_y: Tensor of shape (H, W), real y-component
        vector_field_predicted_x: Tensor of shape (H, W), predicted x-component
        vector_field_predicted_y: Tensor of shape (H, W), predicted y-component

    Returns:
        scaled_error_magnitude: Tensor of shape (H, W), magnitude of the error
                                scaled by the magnitude of the real vector
    """
    # Compute the error vectors
    error_x = vector_field_predicted_x - vector_field_real_x
    error_y = vector_field_predicted_y - vector_field_real_y

    # Magnitude of the error vectors
    error_magnitude = torch.sqrt(error_x ** 2 + error_y ** 2)  # avoid sqrt(0)

    # Magnitude of the real vectors
    real_magnitude = torch.sqrt(vector_field_real_x ** 2 + vector_field_real_y ** 2 )

    # Scale error magnitude by real magnitude
    scaled_error_magnitude = error_magnitude / real_magnitude

    return scaled_error_magnitude