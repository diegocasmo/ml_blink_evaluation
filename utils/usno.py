import numpy as np
from PIL import Image
import skimage.measure
from utils.linalg import normalize

USNO_WIDTH = 297
USNO_HEIGHT = 298
USNO_VECTOR_SIZE = USNO_HEIGHT*USNO_WIDTH

def get_usno_vector(image_key, band, threshold = 60):
  '''
  Return an USNO image specified by the `image_key` and `band` as a vector
  '''
  file_name = 'USNO{}{}.gif'.format(image_key, band)
  file_path = './images/USNO1001/{}'.format(file_name)
  xs = np.asarray(Image.open(file_path)).flatten()
  return np.where(xs > threshold, 255, 0)

def get_usno_projection(image_key, band, num_proj):
  '''
  Return a normalized USNO vector with its dimensionality reduced to `num_proj`
  '''
  xs = get_usno_vector(image_key, band)
  slices = np.arange(0, USNO_VECTOR_SIZE, int(USNO_VECTOR_SIZE / num_proj))
  return normalize(np.add.reduceat(xs, slices))
