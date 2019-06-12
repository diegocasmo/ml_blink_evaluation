import math
from functools import reduce

def l2_norm(xs):
  '''
  Return the L2-norm of the xs vector
  '''
  return math.sqrt(reduce(lambda acc, x: acc + math.pow(x, 2), xs, 0))

def normalize(xs):
  '''
  Return a normalize vector using the L2-norm
  '''
  norm = l2_norm(xs)
  return xs / norm
