from PIL import Image
import numpy as np


def disp_tensor(tens):
    if not isinstance(tens, np.ndarray):
        array = tens.clone().cpu()
        array = array.detach().numpy()
    else:
        array = tens

    if array.ndim > 3:
        array = array[0]
    if array.ndim > 2:
        if array.shape[0] > 3:
            array = array.max(axis=0, keepdims=True)
        elif array.shape[0] == 3:
            array = (array + 1) / 2
    else:
        array = array[None]
    if array.shape[0] == 1:
        array = np.tile(array, (3, 1, 1))
    array = array.transpose((1, 2, 0))
    Image.fromarray((array * 255).astype(np.uint8)).show()
