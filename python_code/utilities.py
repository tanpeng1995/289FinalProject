def normalize_image(img, L):
    import numpy as np
    _min, _max = img.min(), img.max()
    img = ((img-_min)/(_max-_min) * (L-1)).astype(np.uint8)
    return img
