import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    (Conv2d -> InstanceNorm -> LeakyReLU) * 2

    Key Features for Physics-Informed Learning:
    1. Reflect Padding: Prevents 'wall-of-zeros' artifacts at boundaries.
    2. Instance Norm: Stabilizes training for single-image optimization (Expected Batch Size = 1).
    3. LeakyReLU: Preserves negative signals (directionality) crucial for velocity fields.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # --- First Layer ---
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # --- Second Layer ---
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class VectorCombinationUNet(nn.Module):
    """
    Physics-Informed Residual U-Net.

    Goal: Takes an imperfect 'Original' vector field and learns a residual
    correction to satisfy divergence constraints.

    Input:  (Batch, 3, 128, 64) -> [u_original, v_original, mask]
    Output: (Batch, 2, 128, 64) -> [u_final, v_final]
    """

    def __init__(self, n_channels, n_classes):
        super(VectorCombinationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER (Contracting Path) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # --- DECODER (Expansive Path) ---
        # Using Bilinear Upsampling + Conv to avoid checkerboard artifacts

        # Decoder Block 1 (Bottleneck -> Half Res)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=1)
        )
        self.conv_up1 = DoubleConv(256, 128)

        # Decoder Block 2 (Half Res -> Full Res)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        self.conv_up2 = DoubleConv(128, 64)

        # --- OUTPUT LAYER ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # --- ZERO INITIALIZATION ---
        # Forces model to start by predicting "Zero Correction" (Identity mapping)
        with torch.no_grad():
            self.outc.weight.fill_(0)
            self.outc.bias.fill_(0)

    def forward(self, combined_input):
        """
        Args:
            combined_input: Tensor (Batch, 3, 128, 64) containing [u_original, v_original, mask]
        """

        # ===========================
        # 1. ENCODER
        # ===========================

        # Input: (B, 3, 128, 64) -> Output: (B, 64, 128, 64)
        enc1_features = self.inc(combined_input)

        # Input: (B, 64, 128, 64) -> MaxPool -> (B, 64, 64, 32) -> DoubleConv -> Output: (B, 128, 64, 32)
        enc2_features = self.down1(enc1_features)

        # Input: (B, 128, 64, 32) -> MaxPool -> (B, 128, 32, 16) -> DoubleConv -> Output: (B, 256, 32, 16)
        bottleneck_features = self.down2(enc2_features)

        # ===========================
        # 2. DECODER
        # ===========================

        # --- Up 1 ---
        # Upsample Bottleneck: (B, 256, 32, 16) -> (B, 128, 64, 32)
        up1_result = self.up1(bottleneck_features)

        # Concatenate with Encoder 2 features (Skip Connection)
        # (128 ch from up) + (128 ch from enc2) = 256 channels
        merged_features_1 = torch.cat([enc2_features, up1_result], dim=1)

        # Refine Features: Output (B, 128, 64, 32)
        dec1_features = self.conv_up1(merged_features_1)

        # --- Up 2 ---
        # Upsample Dec1: (B, 128, 64, 32) -> (B, 64, 128, 64)
        up2_result = self.up2(dec1_features)

        # Concatenate with Encoder 1 features (Skip Connection)
        # (64 ch from up) + (64 ch from enc1) = 128 channels
        merged_features_2 = torch.cat([enc1_features, up2_result], dim=1)

        # Refine Features: Output (B, 64, 128, 64)
        dec2_features = self.conv_up2(merged_features_2)

        # ===========================
        # 3. PREDICTION & RESIDUAL
        # ===========================

        # Predict the correction field: (B, 2, 128, 64)
        correction_field = self.outc(dec2_features)

        # Extract the original velocity from input (Channels 0 and 1)
        original_velocity = combined_input[:, :2, :, :]
        
        # Extract mask from input (Channels 2 and 3)
        # mask=1 means inpainted region, mask=0 means known region
        mask = combined_input[:, 2:4, :, :]
        
        # CRITICAL: Only apply correction in the INPAINTED region (mask=1)
        # The KNOWN region (mask=0) contains true values and must be preserved exactly
        masked_correction = correction_field * mask

        # Final: V_final = V_original + masked_correction
        # Known region: output = original (correction is zeroed by mask)
        # Inpainted region: output = original + correction
        return original_velocity + masked_correction
