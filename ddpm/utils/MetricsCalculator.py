import torch
import numpy as np
from pathlib import Path

class MetricsCalculator:
    """Calculates error metrics without generating visualizations"""
    
    def __init__(self, results_dir, dd_initializer):
        self.results_dir = Path(results_dir)
        self.dd = dd_initializer
        
    def build_filename(self, prefix, sample_num, noise_type, resamples, num_lines):
        return f"{prefix}{sample_num}_{noise_type}_resample{resamples}_num_lines_{num_lines}.pt"
    
    def load_tensor(self, filename):
        """Load a tensor from a .pt file"""
        try:
            return torch.load(self.results_dir / filename, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return None
    
    def calculate_all_metrics(self, sample_num, noise_type, resamples, num_lines):
        """Calculate all metrics for a given image"""
        
        # Build filenames
        filenames = {
            'ddpm': self.build_filename('ddpm', sample_num, noise_type, resamples, num_lines),
            'initial': self.build_filename('initial', sample_num, noise_type, resamples, num_lines),
            'mask': self.build_filename('mask', sample_num, noise_type, resamples, num_lines),
            'gp_field': self.build_filename('gp_field', sample_num, noise_type, resamples, num_lines)
        }
        
        # Load tensors
        data = {}
        for key, filename in filenames.items():
            tensor = self.load_tensor(filename)
            if tensor is None:
                print(f"Warning: Could not load {key} tensor")
                return {}
            data[key] = tensor
        
        # Extract required tensors
        mask_tensor = data['mask']
        initial_tensor = data['initial']
        
        metrics = {}
        
        # Calculate metrics for each prediction model (ddpm, gp_field)
        for key in ['ddpm', 'gp_field']:
            if key not in data:
                continue
                
            tensor = data[key]
            prefix = f"{key}_"
            
            try:
                metrics[prefix + 'mse'] = self.calculate_mse(
                    tensor, initial_tensor, mask_tensor
                )
                
                metrics[prefix + 'angular_error'] = self.calculate_angular_error(
                    tensor, initial_tensor, mask_tensor
                )
                
                metrics[prefix + 'scaled_error'] = self.calculate_scaled_error(
                    tensor, initial_tensor, mask_tensor
                )
                
                metrics[prefix + 'percent_error'] = self.calculate_percent_error(
                    tensor, initial_tensor, mask_tensor
                )
                
                avg_mag = self.dd.get_attribute(attr='mag_mean')
                
                metrics[prefix + 'magnitude_error'] = self.calculate_magnitude_difference(
                    tensor, initial_tensor, mask_tensor, avg_mag
                )
                
                metrics[prefix + 'magnitude_percent_error'] = self.calculate_magnitude_relative_difference(
                    tensor, initial_tensor, mask_tensor, avg_mag
                )
                
            except Exception as e:
                print(f"Failed to compute metrics for {key}: {e}")
        
        return metrics
    
    def calculate_mse(self, tensor1, tensor2, mask, crop_shape=(44, 94)):
        """Calculate mean squared error"""
        single_mask = mask[:, 0:1, :, :]
        squared_error = (tensor1 - tensor2) ** 2
        pixel_mse = squared_error.sum(dim=1, keepdim=True)
        masked_mse = pixel_mse * single_mask
        
        masked_mse_np = masked_mse.squeeze().cpu().numpy()
        mask_np = single_mask.squeeze().cpu().numpy()
        
        crop_h, crop_w = crop_shape
        cropped_mse = masked_mse_np[:crop_h, :crop_w]
        cropped_mask = mask_np[:crop_h, :crop_w]
        
        valid_pixels = cropped_mask == 1
        avg_mse = cropped_mse[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
        return avg_mse
    
    def calculate_angular_error(self, tensor1, tensor2, mask, crop_shape=(44, 94)):
        """Calculate angular error between vectors"""
        single_mask = mask[:, 0:1, :, :]
        
        u_pred = tensor1[:, 0, :, :]
        v_pred = tensor1[:, 1, :, :]
        u_true = tensor2[:, 0, :, :]
        v_true = tensor2[:, 1, :, :]
        
        dot = u_pred * u_true + v_pred * v_true
        norm_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2) + 1e-8
        norm_true = torch.sqrt(u_true ** 2 + v_true ** 2) + 1e-8
        
        cos_angle = dot / (norm_pred * norm_true)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        angle_deg = angle * (180.0 / np.pi)
        
        angle_np = angle_deg.squeeze().cpu().numpy()
        mask_np = single_mask.squeeze().cpu().numpy()
        
        crop_h, crop_w = crop_shape
        cropped_angle = angle_np[:crop_h, :crop_w]
        cropped_mask = mask_np[:crop_h, :crop_w]
        
        avg_angle_error = cropped_angle[cropped_mask == 1].mean() if np.any(cropped_mask == 1) else float('nan')
        return avg_angle_error
    
    def calculate_scaled_error(self, tensor1, tensor2, mask, crop_shape=(44, 94)):
        """Calculate scaled error magnitude"""
        single_mask = mask[:, 0:1, :, :]
        
        u_pred = tensor1[:, 0, :, :]
        v_pred = tensor1[:, 1, :, :]
        u_true = tensor2[:, 0, :, :]
        v_true = tensor2[:, 1, :, :]
        
        error_u = u_pred - u_true
        error_v = v_pred - v_true
        error_magnitude = torch.sqrt(error_u ** 2 + error_v ** 2)
        real_magnitude = torch.sqrt(u_true ** 2 + v_true ** 2)
        
        scaled_error_magnitude = (error_magnitude / (real_magnitude + 1e-8)) * 100
        masked_scaled_error = scaled_error_magnitude * single_mask.squeeze(1)
        
        masked_scaled_error_np = masked_scaled_error.squeeze().cpu().numpy()
        mask_np = single_mask.squeeze().cpu().numpy()
        
        crop_h, crop_w = crop_shape
        cropped_error = masked_scaled_error_np[:crop_h, :crop_w]
        cropped_mask = mask_np[:crop_h, :crop_w]
        
        valid_pixels = cropped_mask == 1
        avg_scaled_error = cropped_error[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
        return avg_scaled_error
    
    def calculate_percent_error(self, observed, true, mask, crop_shape=(44, 94)):
        """Calculate percent error"""
        single_mask = mask[:, 0:1, :, :]
        
        percent_error = torch.abs((observed - true) / (true + 1e-8)) * 100
        pixel_pe = percent_error.sum(dim=1, keepdim=True)
        masked_per = pixel_pe * single_mask
        
        masked_per_np = masked_per.squeeze().cpu().numpy()
        mask_np = single_mask.squeeze().cpu().numpy()
        
        crop_h, crop_w = crop_shape
        cropped_per = masked_per_np[:crop_h, :crop_w]
        cropped_mask = mask_np[:crop_h, :crop_w]
        
        valid_pixels = cropped_mask == 1
        avg_per = cropped_per[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
        return avg_per
    
    def calculate_magnitude_difference(self, tensor1, tensor2, mask, avg_magnitude, crop_shape=(44, 94)):
        """Calculate normalized magnitude difference"""
        single_mask = mask[:, 0:1, :, :]
        
        mag1 = torch.sqrt(tensor1[:, 0, :, :] ** 2 + tensor1[:, 1, :, :] ** 2)
        mag2 = torch.sqrt(tensor2[:, 0, :, :] ** 2 + tensor2[:, 1, :, :] ** 2)
        
        mag_diff = torch.abs(mag1 - mag2)
        norm_mag_diff = mag_diff / avg_magnitude
        norm_mag_diff = norm_mag_diff * single_mask.squeeze(1)
        
        norm_mag_diff_np = norm_mag_diff.squeeze().cpu().numpy()
        mask_np = single_mask.squeeze().cpu().numpy()
        
        crop_h, crop_w = crop_shape
        cropped_diff = norm_mag_diff_np[:crop_h, :crop_w]
        cropped_mask = mask_np[:crop_h, :crop_w]
        
        cropped_diff = np.flipud(cropped_diff)
        cropped_mask = np.flipud(cropped_mask)
        
        valid_pixels = cropped_mask == 1
        avg_diff = cropped_diff[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
        return avg_diff
    
    def calculate_magnitude_relative_difference(self, tensor1, tensor2, mask, avg_magnitude, crop_shape=(44, 94)):
        """Calculate magnitude percent error"""
        single_mask = mask[:, 0:1, :, :]
        
        mag1 = torch.sqrt(tensor1[:, 0, :, :] ** 2 + tensor1[:, 1, :, :] ** 2)
        mag2 = torch.sqrt(tensor2[:, 0, :, :] ** 2 + tensor2[:, 1, :, :] ** 2)
        
        relative_diff = ((torch.abs(mag1 - mag2)) / (torch.abs(mag2) + 1e-8)) * 100
        relative_mag_diff = relative_diff * single_mask.squeeze(1)
        
        relative_mag_diff_np = relative_mag_diff.squeeze().cpu().numpy()
        mask_np = single_mask.squeeze().cpu().numpy()
        
        crop_h, crop_w = crop_shape
        cropped_diff = relative_mag_diff_np[:crop_h, :crop_w]
        cropped_mask = mask_np[:crop_h, :crop_w]
        
        cropped_diff = np.flipud(cropped_diff)
        cropped_mask = np.flipud(cropped_mask)
        
        valid_pixels = cropped_mask == 1
        avg_diff = cropped_diff[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
        return avg_diff