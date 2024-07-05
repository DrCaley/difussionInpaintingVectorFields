import torch
import torch.nn as nn


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
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half of the network (downsampling path)
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 64, 128), 1, 16),
            MyBlock((16, 64, 128), 16, 16),
            MyBlock((16, 64, 128), 16, 16)
        )
        self.down1 = nn.Conv2d(16, 16, 4, 2, 1)  # (64x128) -> (32x64)

        self.te2 = self._make_te(time_emb_dim, 16)
        self.b2 = nn.Sequential(
            MyBlock((16, 32, 64), 16, 32),
            MyBlock((32, 32, 64), 32, 32),
            MyBlock((32, 32, 64), 32, 32)
        )
        self.down2 = nn.Conv2d(32, 32, 4, 2, 1)  # (32x64) -> (16x32)

        self.te3 = self._make_te(time_emb_dim, 32)
        self.b3 = nn.Sequential(
            MyBlock((32, 16, 32), 32, 64),
            MyBlock((64, 16, 32), 64, 64),
            MyBlock((64, 16, 32), 64, 64)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),  # (16x32) -> (8x16)
            nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1)  # (8x16) -> (4x8)
        )

        # Bottleneck (middle part of the network)
        self.te_mid = self._make_te(time_emb_dim, 64)
        self.b_mid = nn.Sequential(
            MyBlock((64, 4, 8), 64, 128),
            MyBlock((128, 4, 8), 128, 128),
            MyBlock((128, 4, 8), 128, 64)
        )

        # Second half of the network (upsampling path)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (4x8) -> (8x16)
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1)  # (8x16) -> (16x32)
        )

        self.te4 = self._make_te(time_emb_dim, 128)
        self.b4 = nn.Sequential(
            MyBlock((128, 16, 32), 128, 64),
            MyBlock((64, 16, 32), 64, 32),
            MyBlock((32, 16, 32), 32, 32)
        )

        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # (16x32) -> (32x64)
        self.te5 = self._make_te(time_emb_dim, 64)
        self.b5 = nn.Sequential(
            MyBlock((64, 32, 64), 64, 32),
            MyBlock((32, 32, 64), 32, 16),
            MyBlock((16, 32, 64), 16, 16)
        )

        self.up3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)  # (32x64) -> (64x128)
        self.te_out = self._make_te(time_emb_dim, 32)
        self.b_out = nn.Sequential(
            MyBlock((32, 64, 128), 32, 16),
            MyBlock((16, 64, 128), 16, 16),
            MyBlock((16, 64, 128), 16, 16, normalize=False)
        )

        self.conv_out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x, t):
        # Get the time embedding
        t = self.time_embed(t)
        n = len(x)

        # First downsampling block
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 16, 64, 128)

        # Second downsampling block
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 32, 32, 64)

        # Third downsampling block
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 64, 16, 32)

        # Bottleneck
        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 128, 4, 8)

        # First upsampling block
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 128, 16, 32)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 64, 16, 32)

        # Second upsampling block
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 64, 32, 64)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 32, 32, 64)

        # Third upsampling block
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 32, 64, 128)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 16, 64, 128)

        # Final convolution to get the output
        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        # Helper function to create a time embedding MLP
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )