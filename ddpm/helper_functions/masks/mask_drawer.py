import tkinter as tk
import numpy as np
import torch
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from data_prep.data_initializer import DDInitializer
from collections import deque

class ManualMaskDrawer(MaskGenerator):
    def __init__(self, height=44, width=94, pixel_size=8):
        self.h = height
        self.w = width
        self.pixel_size = pixel_size
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)

        self.root = tk.Tk()
        self.root.title("🎨 Mask Drawer - Left: Draw | Right: Erase | Middle: Fill | Shift+Middle: Erase Fill | s: Save | c: Clear")
        self.canvas = tk.Canvas(self.root, width=self.w * pixel_size, height=self.h * pixel_size, bg='white')
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<B3-Motion>", self._erase)
        self.canvas.bind("<Button-2>", self._handle_middle_click)
        self.root.bind("s", self._save_and_close)
        self.root.bind("c", self._clear)

        self.rect_refs = [[None for _ in range(self.w)] for _ in range(self.h)]
        self._draw_initial_grid()

        print("🖌️ Use mouse to draw mask. Press 's' to save, 'c' to clear.")
        print("Black = Mask, White = Prediction")

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

    def _handle_middle_click(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if event.state & 0x0001:  # Shift key mask
            self._flood_fill(x, y, erase=True)
        else:
            self._flood_fill(x, y, erase=False)

    def _flood_fill(self, x, y, erase=False):
        target = 1 if erase else 0
        replace = 0 if erase else 1
        fill_color = 'white' if erase else 'black'

        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        if self.mask[y, x] != target:
            return

        queue = deque()
        queue.append((x, y))

        while queue:
            cx, cy = queue.popleft()
            if 0 <= cx < self.w and 0 <= cy < self.h and self.mask[cy, cx] == target:
                self.mask[cy, cx] = replace
                self.canvas.itemconfig(self.rect_refs[cy][cx], fill=fill_color)
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    queue.append((cx+dx, cy+dy))

    def _clear(self, event=None):
        self.mask[:, :] = 0
        for y in range(self.h):
            for x in range(self.w):
                self.canvas.itemconfig(self.rect_refs[y][x], fill='white')
        print("🧽 Canvas cleared.")

    def _save_and_close(self, event=None):
        print("✅ Mask saved and window closed.")
        self.root.destroy()

    def generate_mask(self, image_shape=None):
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

        flipped_mask = torch.flip(self.tensor_mask, dims=[-2])

        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        padded_mask = torch.nn.functional.pad(flipped_mask,
                                              (0, pad_W, 0, pad_H),
                                              mode='constant', value=0)

        return 1.0 - padded_mask

    def __str__(self):
        return "ManualMaskDrawer"

    def get_num_lines(self):
        return int(np.sum(self.mask))
