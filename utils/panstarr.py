import numpy as np
from PIL import Image
import skimage.measure
from utils.linalg import normalize

PANSTARR_WIDTH = 1200
PANSTARR_HEIGHT = PANSTARR_WIDTH
PANSTARR_VECTOR_SIZE = PANSTARR_HEIGHT*PANSTARR_WIDTH

def get_panstarr_vector(image_key, band, threshold = 220):
  '''
  Return a PanSTARRs image specified by the `image_key` and `band` as a vector
  '''
  file_name = 'PanSTARR{}{}.jpg'.format(image_key, band)
  file_path = './images/PanSTARRS_ltd/{}'.format(file_name)
  xs = np.asarray(Image.open(file_path)).flatten()
  return np.where(xs > threshold, 255, 0)

def get_panstarr_projection(image_key, band, num_proj):
  '''
  Return a normalized PanSTARRs vector with its dimensionality reduced to `num_proj`
  '''
  xs = get_panstarr_vector(image_key, band)
  slices = np.arange(0, PANSTARR_VECTOR_SIZE, int(PANSTARR_VECTOR_SIZE / num_proj))
  return normalize(np.add.reduceat(xs, slices))
