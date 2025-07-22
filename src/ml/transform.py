import numpy as np
import torch
import torch.nn.functional as F

from src.ml.utils import reshape_data
from src.settings.config import ebno_db, num_bs_ant, num_ut_ant, speed
from src.settings.ml import number_of_samples

# Assuming these are defined elsewhere in your project
from src.utils.plots import plot_symbols


class MinMaxScaler4D:
    def __init__(self, feature_range=(-1, 1), device="cpu"):
        """
        Args:
            feature_range (tuple or None): Desired range for normalization, default (-1, 1).
                                          If None, no normalization is applied.
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.feature_range = feature_range  # Can be None to skip normalization
        self.device = torch.device(device)
        self.original_shape = None
        self.transformed_shape = None
        self.shape_list = [8, 16, 32, 64, 128, 256, 512, 1024]
        self.min_val = None
        self.max_val = None
        self.residual = None  # Store residual for accurate recovery

    def _to_tensor(self, data):
        """Convert data to PyTorch tensor"""
        if isinstance(data, np.ndarray):
            return torch.FloatTensor(data).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise ValueError("Input must be numpy array or torch tensor")

    def _to_numpy(self, data):
        """Convert tensor to numpy"""
        return data.cpu().numpy()

    def _find_closest_above(self, value):
        """Find the smallest value in shape_list that is >= input value"""
        for shape in self.shape_list:
            if shape >= value:
                return shape
        return self.shape_list[-1]

    def _resize(self, data, target_h, target_w):
        """Resize data using bicubic interpolation"""
        if len(data.shape) == 4:
            return F.interpolate(
                data, size=(target_h, target_w), mode="bicubic", align_corners=True
            )
        else:
            data = data.unsqueeze(0)
            resized = F.interpolate(
                data, size=(target_h, target_w), mode="bicubic", align_corners=True
            )
            return resized.squeeze(0)

    def fit_transform(self, data, return_numpy=True):
        """Transform data with optional normalization and resizing, storing residual"""
        data = self._to_tensor(data)
        self.original_shape = data.shape
        h, w = self.original_shape[-2:]

        # Normalize only if feature_range is specified
        if self.feature_range is not None:
            self.min_val = torch.min(data)
            self.max_val = torch.max(data)
            if self.max_val != self.min_val:
                normalized = (data - self.min_val) / (self.max_val - self.min_val) * (
                    self.feature_range[1] - self.feature_range[0]
                ) + self.feature_range[0]
            else:
                normalized = data * 0 + self.feature_range[0]
        else:
            normalized = data  # Skip normalization

        # Resize to target shape
        h_target = self._find_closest_above(h)
        w_target = self._find_closest_above(w)
        transformed = self._resize(normalized, h_target, w_target)
        self.transformed_shape = transformed.shape

        # Compute residual for accurate recovery
        transformed_back = self._resize(transformed, h, w)
        self.residual = normalized - transformed_back  # Error from resizing

        if return_numpy:
            transformed = self._to_numpy(transformed)
        return transformed

    def inverse_transform(self, transformed_data, return_numpy=True):
        """Reverse the transformation with residual correction"""
        transformed_data = self._to_tensor(transformed_data)

        if self.original_shape is None or self.transformed_shape is None:
            raise ValueError("Scaler must be fitted first using fit_transform.")

        orig_h, orig_w = self.original_shape[-2:]

        # Resize back to original shape
        resized_back = self._resize(transformed_data, orig_h, orig_w)

        # Apply residual correction
        if self.residual is not None:
            corrected = resized_back + self.residual
        else:
            corrected = resized_back

        # Denormalize only if feature_range was specified
        if self.feature_range is not None:
            if self.max_val != self.min_val:
                recovered = (corrected - self.feature_range[0]) / (
                    self.feature_range[1] - self.feature_range[0]
                ) * (self.max_val - self.min_val) + self.min_val
            else:
                recovered = corrected * 0 + self.min_val
        else:
            recovered = corrected  # No denormalization

        if return_numpy:
            recovered = self._to_numpy(recovered)
        return recovered


# Example usage
if __name__ == "__main__":
    save_dir = f"txant_{num_ut_ant}_rxant_{num_bs_ant}_speed_{speed}_samples_{number_of_samples}_ebno_{ebno_db}"
    pilot_matrices = np.load(f"data/{save_dir}/pilot_matrices.npy")[:100]
    h_freqs = np.load(f"data/{save_dir}/h_freqs.npy")[:100]

    pilot_matrices = reshape_data(pilot_matrices)
    h_freqs = reshape_data(h_freqs)

    data = h_freqs

    # Test with normalization
    scaler_with_norm = MinMaxScaler4D(feature_range=(-1, 1), device="cpu")
    transformed_data_norm = scaler_with_norm.fit_transform(data, return_numpy=True)
    # transformed_data_norm += np.random.normal(loc=0, scale=np.sqrt(noise_variance / 2), size=transformed_data_norm.shape)
    recovered_data_norm = scaler_with_norm.inverse_transform(
        transformed_data_norm, return_numpy=True
    )

    print("With Normalization:")
    print("Original shape:", data.shape)
    print("Transformed shape:", transformed_data_norm.shape)
    print("Original range:", data.min(), data.max())
    print(
        "Transformed range:", transformed_data_norm.min(), transformed_data_norm.max()
    )
    print("Recovered shape:", recovered_data_norm.shape)
    print("Max absolute difference:", np.max(np.abs(data - recovered_data_norm)))
    print("Mean absolute difference:", np.mean(np.abs(data - recovered_data_norm)))

    # Plotting (optional)
    plot_symbols(symbol=data[0, 0].T)
    plot_symbols(symbol=transformed_data_norm[0, 0].T)
    plot_symbols(symbol=recovered_data_norm[0, 0].T)
