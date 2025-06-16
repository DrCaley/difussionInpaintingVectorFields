import tkinter as tk
import numpy as np
import torch
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from data_prep.data_initializer import DDInitializer

class ManualMaskDrawer(MaskGenerator):
    def __init__(self, height=44, width=94, pixel_size=8):
        self.h = height
        self.w = width
        self.pixel_size = pixel_size
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)

        self.root = tk.Tk()
        self.root.title("🎨 Mask Drawer - Left: Draw | Right: Erase | s: Save | c: Clear")
        self.canvas = tk.Canvas(self.root, width=self.w * pixel_size, height=self.h * pixel_size, bg='white')
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<B3-Motion>", self._erase)
        self.root.bind("s", self._save_and_close)
        self.root.bind("c", self._clear)

        self.rect_refs = [[None for _ in range(self.w)] for _ in range(self.h)]

        self._draw_initial_grid()
        print("🖌️ Use mouse to draw mask. Press 's' to save, 'c' to clear.")
        self.root.mainloop()

        dd = DDInitializer()
        base_mask = torch.tensor(self.mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        self.tensor_mask = base_mask.repeat(2, 1, 1)  # (2, H, W)
        self.tensor_mask = self.tensor_mask.unsqueeze(0).to(dd.get_device())  # (1, 2, H, W)

    def _draw_initial_grid(self):
        for y in range(self.h):
            for x in range(self.w):
                px = x * self.pixel_size
                py = y * self.pixel_size
                rect = self.canvas.create_rectangle(px, py, px + self.pixel_size, py + self.pixel_size,
                                                    fill='white', outline='gray')
                self.rect_refs[y][x] = rect

    def _draw(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= x < self.w and 0 <= y < self.h:
            self.mask[y, x] = 1
            self.canvas.itemconfig(self.rect_refs[y][x], fill='black')

    def _erase(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= x < self.w and 0 <= y < self.h:
            self.mask[y, x] = 0
            self.canvas.itemconfig(self.rect_refs[y][x], fill='white')

    def _clear(self, event=None):
        self.mask[:, :] = 0
        for y in range(self.h):
            for x in range(self.w):
                self.canvas.itemconfig(self.rect_refs[y][x], fill='white')
        print("🧽 Canvas cleared.")

    def _save_and_close(self, event=None):
        print("✅ Mask saved and window closed.")
        self.root.destroy()

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            return self.tensor_mask

        if isinstance(image_shape, torch.Size):
            image_shape = tuple(image_shape)

        if len(image_shape) != 4 or image_shape[0] != 1 or image_shape[1] != 2:
            raise ValueError(f"Expected image shape (1, 2, H, W), got {image_shape}")

        _, _, H, W = image_shape
        mask_H, mask_W = self.tensor_mask.shape[-2:]

        if mask_H > H or mask_W > W:
            raise ValueError(f"Drawn mask is larger than the image. "
                             f"Image: ({H}, {W}), Mask: ({mask_H}, {mask_W})")

        pad_H = H - mask_H
        pad_W = W - mask_W

        # Pad bottom and right (top-left aligned)
        padded_mask = torch.nn.functional.pad(self.tensor_mask,
                                              (0, pad_W, 0, pad_H),  # (left, right, top, bottom)
                                              mode='constant', value=0)
        return padded_mask

    def __str__(self):
        return "ManualMaskDrawer"

    def get_num_lines(self):
        return int(np.sum(self.mask))
