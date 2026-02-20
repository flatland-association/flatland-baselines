from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def main(cells_1: List[Tuple[int, int]], cells_2: List[Tuple[int, int]]):
    height = max([r for r, c in cells_1] + [r for r, c in cells_2]) + 1
    width = max([c for r, c in cells_1] + [c for r, c in cells_2]) + 1
    im1 = np.zeros((height, width))
    for cell in cells_1:
        im1[cell] = 1
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    im2 = np.zeros((height, width))
    for cell in cells_2:
        im2[cell] = 1
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.show()


if __name__ == '__main__':
    # copy from output
    cells_1 = []
    cells_2 = []

    main(cells_1, cells_2)
