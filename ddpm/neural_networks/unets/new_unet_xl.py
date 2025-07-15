import torch
import torch.nn as nn

from ddpm.neural_networks.unets.net import PConv2d, PTranspose2d

"""The framework for the xl model, currently the best as of Feb 2025"""

def sinusoidal_embedding(n, d):
    """
    Generates a sinusoidal embedding for each time-step.
    This function maps each scalar time-step to a higher-dimensional vector
    using sinusoidal functions.

    :param n: Number of time-steps
    :param d: Dimensionality of the embedding
    :return: Tensor of shape (n, d) containing the sinusoidal embeddings.
    """
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, 1::2])

    return embedding

class my_block(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(my_block, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = PConv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = PConv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x, mask):
        out = self.ln(x) if self.normalize else x
        out, mask = self.conv1(out, mask)
        out = self.activation(out)
        out, mask = self.conv2(out, mask)
        out = self.activation(out)
        return out, mask


class SequentialWithMask(nn.Sequential):
    def forward(self, x, mask=None):
        for module in self:
            x, mask = module(x, mask)
        return x, mask

class MaskedActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, mask):
        return self.activation(x), mask


class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half of the network (down-sampling path)
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = SequentialWithMask(
            my_block((2, 64, 128), 2, 16),
            my_block((16, 64, 128), 16, 16),
            my_block((16, 64, 128), 16, 16)
        )
        # (64x128) -> (32x64)
        self.down1 = PConv2d(16, 16, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 16)
        self.b2 = SequentialWithMask(
            my_block((16, 32, 64), 16, 32),
            my_block((32, 32, 64), 32, 32),
            my_block((32, 32, 64), 32, 32)
        )
        # (32x64) -> (16x32)
        self.down2 = PConv2d(32, 32, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 32)
        self.b3 = SequentialWithMask(
            my_block((32, 16, 32), 32, 64),
            my_block((64, 16, 32), 64, 64),
            my_block((64, 16, 32), 64, 64)
        )
        # (16x32) -> (8x16)
        self.down3 = PConv2d(64, 64, 4, 2, 1)

        self.te4 = self._make_te(time_emb_dim, 64)
        self.b4 = SequentialWithMask(
            my_block((64, 8, 16), 64, 128),
            my_block((128, 8, 16), 128, 128),
            my_block((128, 8, 16), 128, 128)
        )
        self.down4 = SequentialWithMask(
            # (8x16) -> (4x8)
            PConv2d(128, 128, 4, 2, 1),
            MaskedActivation(nn.SiLU()),
            # (4x8) -> (2x4)
            PConv2d(128, 128, 4, 2, 1)
        )

        # Bottleneck (middle part of the network)
        self.te_mid = self._make_te(time_emb_dim, 128)
        self.b_mid = SequentialWithMask(
            my_block((128, 2, 4), 128, 256),
            my_block((256, 2, 4), 256, 256),
            my_block((256, 2, 4), 256, 128)
        )

        # Second half of the network (upsampling path)
        self.up1 = SequentialWithMask(
            # (2x4) -> (4x8)
            PTranspose2d(128, 128, 4, 2, 1),
            MaskedActivation(nn.SiLU()),
            # (4x8) -> (8x16)
            PTranspose2d(128, 128, 4, 2, 1)
        )

        self.te5 = self._make_te(time_emb_dim, 256)
        self.b5 = SequentialWithMask(
            my_block((256, 8, 16), 256, 128),
            my_block((128, 8, 16), 128, 64),
            my_block((64, 8, 16), 64, 64)
        )
        # (8x16) -> (16x32)
        self.up2 = PTranspose2d(64, 64, 4, 2, 1)
        self.te6 = self._make_te(time_emb_dim, 128)
        self.b6 = SequentialWithMask(
            my_block((128, 16, 32), 128, 64),
            my_block((64, 16, 32), 64, 32),
            my_block((32, 16, 32), 32, 32)
        )
        # (16x32) -> (32x64)
        self.up3 = PTranspose2d(32, 32, 4, 2, 1)
        self.te7 = self._make_te(time_emb_dim, 64)
        self.b7 = SequentialWithMask(
            my_block((64, 32, 64), 64, 32),
            my_block((32, 32, 64), 32, 16),
            my_block((16, 32, 64), 16, 16)
        )
        # (32x64) -> (64x128)
        self.up4 = PTranspose2d(16, 16, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 32)
        self.b_out = SequentialWithMask(
            my_block((32, 64, 128), 32, 16),
            my_block((16, 64, 128), 16, 16),
            my_block((16, 64, 128), 16, 16, normalize=False)
        )

        self.conv_out = PConv2d(16, 2, 3, 1, 1)

    def forward(self, x, t, mask):
        # Get the time embedding
        t = self.time_embed(t)
        n = len(x)

        # First downsampling block
        out1, mask1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1), mask=mask)  # (N, 16, 64, 128)

        # Second downsampling block
        x_down1, mask_down1 = self.down1(out1, mask1)
        out2, mask2 = self.b2(x_down1 + self.te2(t).reshape(n, -1, 1, 1), mask=mask_down1)  # (N, 32, 32, 64)

        # Third downsampling block
        x_down2, mask_down2 = self.down2(out2, mask2)
        out3, mask3 = self.b3(x_down2 + self.te3(t).reshape(n, -1, 1, 1), mask=mask_down2)  # (N, 64, 16, 32)

        # Fourth downsampling block
        x_down3, mask_down3 = self.down3(out3, mask3)
        out4, mask4 = self.b4(x_down3 + self.te4(t).reshape(n, -1, 1, 1), mask=mask_down3)  # (N, 128, 8, 16)

        # Bottleneck
        x_down4, mask_down4 = self.down4(out4, mask4)
        out_mid, mask_mid = self.b_mid(x_down4 + self.te_mid(t).reshape(n, -1, 1, 1), mask_down4)  # (N, 128, 2, 4)

        # First upsampling block
        up1_out, up1_mask = self.up1(out_mid, mask_mid)  # (N, 128, 8, 16)
        out5 = torch.cat((out4, up1_out), dim=1)  # (N, 256, 8, 16)
        mask5 = torch.cat((mask4, up1_mask), dim=1)  # (N, 256, 8, 16)
        out5, mask5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1), mask=mask5)  # (N, 64, 8, 16)

        # Second upsampling block
        up2_out, up2_mask = self.up2(out5, mask5)  # (N, 64, 16, 32)
        out6 = torch.cat((out3, up2_out), dim=1)  # (N, 128, 16, 32)
        mask6 = torch.cat((mask3, up2_mask), dim=1)  # (N, 128, 16, 32)
        out6, mask6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1), mask=mask6)  # (N, 32, 16, 32)

        # Third upsampling block
        up3_out, up3_mask = self.up3(out6, mask6)  # (N, 32, 32, 64)
        out7 = torch.cat((out2, up3_out), dim=1)  # (N, 64, 32, 64)
        mask7 = torch.cat((mask2, up3_mask), dim=1)  # (N, 64, 32, 64)
        out7, mask7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1), mask=mask7)  # (N, 16, 32, 64)

        # Fourth upsampling block
        up4_out, up4_mask = self.up4(out7, mask7)  # (N, 16, 64, 128)
        out = torch.cat((out1, up4_out), dim=1)  # (N, 32, 64, 128)
        final_mask = torch.cat((mask1, up4_mask), dim=1)  # (N, 32, 64, 128)
        out, final_mask = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1), mask=final_mask)  # (N, 16, 64, 128)

        # Final convolution to get the output
        out, final_mask = self.conv_out(out, final_mask)

        return out, final_mask

    def _make_te(self, dim_in, dim_out):
        # Helper function to create a time embedding MLP
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
