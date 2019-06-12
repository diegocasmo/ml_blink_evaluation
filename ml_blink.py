import sys
import random
import operator
import numpy as np
from functools import reduce
from datetime import datetime
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from utils.dataset_bands import datasets_bands
from utils.usno import get_usno_projection, get_usno_vector
from utils.panstarr import get_panstarr_projection, get_panstarr_vector

def generate_candidates(m, bands):
  '''
  Generate candidates for each element in m for each dataset band
  '''
  return reduce(operator.concat, [
    [
      {
        'image_key': i,
        'usno_band': bands[j].get('USNO'),
        'panstarr_band': bands[j].get('PanSTARR')
      }
      for j in range(len(bands))
    ]
    for i in range(m)
  ])

def get_v_cid(s):
  '''
  Return a string which uniquely identifies an element of `S`
  '''
  return '{}.{}.{}'.format(s.get('image_key'), s.get('usno_band'), s.get('panstarr_band'))

def get_anomalies():
  '''
  Return mission's `v_cid` of those known to be anomalies
  '''
  return [
    get_v_cid({
      'image_key': 13,
      'usno_band': 'blue1',
      'panstarr_band': 'g'
    }),
    get_v_cid({
      'image_key': 13,
      'usno_band': 'blue2',
      'panstarr_band': 'g'
    }),
    get_v_cid({
      'image_key': 56,
      'usno_band': 'blue1',
      'panstarr_band': 'g'
    }),
    get_v_cid({
      'image_key': 56,
      'usno_band': 'blue2',
      'panstarr_band': 'g'
    }),
    get_v_cid({
      'image_key': 679,
      'usno_band': 'ir',
      'panstarr_band': 'z'
    }),
    get_v_cid({
      'image_key': 831,
      'usno_band': 'red1',
      'panstarr_band': 'r'
    }),
    get_v_cid({
      'image_key': 831,
      'usno_band': 'red2',
      'panstarr_band': 'r'
    })
  ]

def compute_vs(S, A):
  '''
  Compute the value `v` for all `s` in `S` using the active set `A`
  '''
  # Compute v for each element in S
  vs = {}
  for s in S:
    # Each v is initially set to 0
    v_cid = get_v_cid(s)
    vs[v_cid] = 0

    # Use projection to reduce the dimensionality of x and y
    x = s.get('usno_vector')
    y = s.get('panstarr_vector')

    for member in A:
      xi = member.get('usno_vector')
      yi = member.get('panstarr_vector')

      # Compute `v`
      v = np.dot(np.dot(x, xi), np.dot(y, yi))

      # Keep track of each `v` value using `cid`
      vs[v_cid] = vs[v_cid] + v if v_cid in vs else v

  return {k: v for k, v in vs.items() if v >= 0}

def compute_roc_curve(S, A):
  '''
  Compute the ROC Curve in different steps of threshold `v`
  '''
  # Compute `v` value for the entire dataset
  vs = compute_vs(S, A)
  anomalies = get_anomalies()

  # Number of real anomalies in the data (p = positives)
  p = len(anomalies)
  # Real number of non-anomalies in the data (n = negatives)
  n = len(S) - p

  fpr, tpr = [], []
  for v in sorted(vs.values()):
    potential_anomalies = list(filter(lambda x: vs[x] < v, vs))

    # How many < `v` are correctly classified as anomaly
    tp = reduce(lambda acc, x: acc + 1 if x in anomalies else acc, potential_anomalies, 0)
    # How many < `v` are incorrectly classified as anomaly
    fp = len(potential_anomalies) - tp

    # Compute false positive rate (fpr) and true positive rate (tpr)
    fpr.append(fp / float(n))
    tpr.append(tp / float(p))

  return (fpr, tpr)

def export_roc_curve_plot(S, A, t, num_proj, num_anomalies_found):
  '''
  Plot ROC curve of `S`, using the active set `A`
  '''
  fpr, tpr = compute_roc_curve(S, A)
  plt.plot(fpr, tpr, label='t = {}, AUC = {}'.format(t, '{}'.format(auc(fpr, tpr))[0:4]))
  plt.title('ROC Curve')
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.legend()
  plt.savefig('./plots/{}_anomalies_t_{}.png'.format(num_anomalies_found, t))

def tcrawl_candidates(num_proj, max_time_steps):
  '''
  Crawl candidates and recommend those evaluated to be anomalies
  '''
  try:
    # Generate candidates that will be crawled
    m = 1001
    roc_delimiter = 50
    vs_delimiter = 1

    # Retrieve anomalies in the dataset
    anomalies = get_anomalies()

    # Store pre-processed S in memory for faster processing
    S = [
      {
        'image_key': s.get('image_key'),
        'usno_band': s.get('usno_band'),
        'panstarr_band': s.get('panstarr_band'),
        'usno_vector': get_usno_projection(s.get('image_key'), s.get('usno_band'), num_proj),
        'panstarr_vector': get_panstarr_projection(s.get('image_key'), s.get('panstarr_band'), num_proj)
      }
      for s in generate_candidates(m, datasets_bands)
    ]

    # Create a copy with all elements of `S` to use for computing the ROC curve
    S_original = S.copy()

    # Capture `v` values of missions across different time steps
    ts_agg = []
    vs_min_agg = []
    anomalies_found = []

    anomalies_agg = {}
    for x in anomalies:
      anomalies_agg[x] = []

    normal_agg = {
      '0.blue1.g': []
    }

    t = 0
    A = []
    stop_criteria = False
    while not stop_criteria:
      # Find minimum `v` value and index
      vs = compute_vs(S, A)
      vm = min(vs.values())
      vm_cid = [k for k in vs if vs[k] == vm]
      # Break ties randomly
      if len(vm_cid) > 1:
        vm_cid = random.choice(vm_cid)
      else:
        vm_cid = vm_cid[0]

      # Remove anomaly from S if found, add to the active set otherwise
      if vm_cid in anomalies:
        S = list(filter(lambda x: get_v_cid(x) != vm_cid, S))
        anomalies_found.append((vm_cid, t))
      else:
        x = list(filter(lambda x: get_v_cid(x) == vm_cid, S))[0]
        A.append(x)

      # Verify if all anomalies have been found before max time steps are reached
      early_stop = (len(anomalies_found) == len(anomalies) and t < max_time_steps)

      # Export ROC curve every `roc_delimiter`
      if (t % roc_delimiter == 0 and t > 0) or early_stop:
        export_roc_curve_plot(S_original, A, t, num_proj, len(anomalies_found))

      # Aggregate `v` values every `vs_delimiter`
      if t % vs_delimiter == 0 or early_stop:
        vs = compute_vs(S_original, A)
        ts_agg.append(t)
        vs_min_agg.append(vm)
        for x in anomalies_agg:
          anomalies_agg[x].append(vs[x])
        for x in normal_agg:
          normal_agg[x].append(vs[x])

      print('[t vm_cid vm len(S) len(A)]: [{} {} {} {} {}]'.format(t, vm_cid, vm, len(S), len(A)))
      t = t + 1
      stop_criteria = len(anomalies_found) == len(anomalies) or t == max_time_steps

    # Create plot of anomalies found
    plt.figure(100000)
    plt.plot(ts_agg, vs_min_agg, label='min(v)')

    # Plot anomalies
    for (vm_cid, t_found) in anomalies_found:
      plt.plot(ts_agg, anomalies_agg[vm_cid], label='Anomaly ({}), t = {}'.format(vm_cid, t_found))

    # Plot normal observations
    for x in normal_agg:
      plt.plot(ts_agg, normal_agg[x], label='Normal ({})'.format(x))

    plt.title('Anomalies Found')
    plt.xlabel('Time steps')
    plt.ylabel('v')
    plt.legend()
    plt.savefig('./plots/found_{}_anomalies_t_{}.png'.format(len(anomalies_found), t - 1))

    # Create plot of anomalies not found
    plt.figure(200000)
    plt.plot(ts_agg, vs_min_agg, label='min(v)')

    # Plot anomalies not found
    for x in [x for x in anomalies if x not in list(map(lambda x: x[0], anomalies_found))]:
      plt.plot(ts_agg, anomalies_agg[x], label='Anomaly ({})'.format(x))

    # Plot normal observations
    for x in normal_agg:
      plt.plot(ts_agg, normal_agg[x], label='Normal ({})'.format(x))

    plt.title('Anomalies Not Found')
    plt.xlabel('Time steps')
    plt.ylabel('v')
    plt.legend()
    plt.savefig('./plots/not_found_{}_anomalies_t_{}.png'.format(len(anomalies) - len(anomalies_found), t - 1))

    # Print summary
    print('Found: {}/{} anomalies'.format(len(anomalies_found), len(anomalies)))
  except Exception as e:
    print('******Unable to crawl candidates: {}******'.format(e))

if __name__ == '__main__':
  start_time = datetime.now()
  num_proj, max_time_steps = int(sys.argv[1]), int(sys.argv[2])
  print('---Crawling candidates using {} projections up to {} time steps---\n'.format(num_proj, max_time_steps))
  tcrawl_candidates(num_proj, max_time_steps)
  print('Elapsed time: {}'.format(datetime.now() - start_time))
