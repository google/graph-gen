# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Credits for this code go to Yujia Li, Research Scientist at Deepmind. Thanks
   Yujia, for writing a general purpose script for parsing molecule datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

CHEMBL_DATA_ROOT = '/usr/local/aviralkumar/Desktop/code/graph-gen/research/graph_gen/data/chembl/processed/'
ZINC_DATA_ROOT = '/usr/local/aviralkumar/Desktop/code/graph-gen/research/graph_gen/data/zinc_drugs/processed/'
COND_DATA_ROOT = CHEMBL_DATA_ROOT

flags = tf.app.flags
FLAGS = flags.FLAGS

class Dataset(object):
  def __init__(self, data, batch_size, shuffle=True):
    """Pass the batch size as an integer and if
      shuffle is set to true the dataset will be randomly shuffled.
    """
    self.data = data
    self.batch_size = batch_size
    self.shuffle = shuffle
    self._n = len(data)
    self.reset()

  def reset(self):
    self._idx = (np.random.permutation(self._n) if self.shuffle
                 else range(self._n))
    self._i_next = 0

  def next(self):
    batch = []
    while len(batch) < self.batch_size:
      batch.append(self.data[self._idx[self._i_next]])
      self._i_next += 1
      if self._i_next == self._n:
        self.reset()
    return batch

def read_dataset(dataset):
  train_str = 'train'
  val_str = 'val'
  test_str = 'test'
  if dataset.startswith('zinc'):
    path = ZINC_DATA_ROOT
    train_str = train_str + '.smi'
    val_str = val_str + '.smi'
    test_str = test_str + '.smi'
  elif dataset.startswith('cond'):
    path = COND_DATA_ROOT+dataset+'.'
    test_str = None
  else:
    train_set = read_strings_from_file(CHEMBL_DATA_ROOT + dataset + '.train')
    if 'sub' in dataset:
      loc = dataset.rfind('sub')
      full_set = dataset[:loc]
    else:
      full_set = dataset
    val_set = read_strings_from_file(CHEMBL_DATA_ROOT + full_set + '.val')
    test_set = read_strings_from_file(CHEMBL_DATA_ROOT + full_set + '.test')
    return train_set, val_set, test_set

  train_set = read_strings_from_file(path + train_str)
  val_set = read_strings_from_file(path + val_str)
  if test_str is not None:
    test_set = read_strings_from_file(path + test_str)
  else:
    test_set = None
  return train_set, val_set, test_set

def read_strings_from_file(file_path):
  with tf.gfile.GFile(file_path) as f:
    return [l.strip() for l in f.readlines()]

def read_molecules_from_file(file_path):
  mols = []
  with tf.gfile.GFile(file_path) as f:
    m = get_molecule_from_file(f)
    while m is not None:
      mols.append(m)
      m = get_molecule_from_file(f)
  return mols

def read_molecule_graphs_set(dataset):
  """Read the graphs for a specified dataset."""
  train_str = 'train'
  val_str = 'val'
  test_str = 'test'
  if dataset.startswith('zinc'):
    path = ZINC_DATA_ROOT
    train_str = train_str + '.graphs'
    val_str = val_str + '.graphs'
    test_str = test_str + '.graphs'
  elif dataset.startswith('cond'):
    path = COND_DATA_ROOT+dataset+'.'
    test_str = None
    train_str = '_graph.' + train_str
    val_str = '_graph.' + val_str
  else:
    train_set = read_molecules_from_file(
        CHEMBL_DATA_ROOT + dataset + '_graph.train')
    if 'sub' in dataset:
      loc = dataset.rfind('sub')
      full_set = dataset[:loc]
    else:
      full_set = dataset
    val_set = read_molecules_from_file(
        CHEMBL_DATA_ROOT + full_set + '_graph.val')
    test_set = read_molecules_from_file(
        CHEMBL_DATA_ROOT + full_set + '_graph.test')
    return train_set, val_set, test_set

  train_set = read_molecules_from_file(path + train_str)
  val_set = read_molecules_from_file(path + val_str)
  if test_str is None:
    test_set = None
  else:
    test_set = read_molecules_from_file(path + test_str)
  return train_set, val_set, test_set

def read_molecule_mapping_from_freq_table(file_path):
  """indxe -> symbol."""
  with tf.gfile.GFile(file_path) as f:
    return [line.split(':')[0] for line in f.readlines()]

def read_molecule_mapping_for_set(dataset):
  if 'sub' in dataset:
    loc = dataset.rfind('sub')
    dataset = dataset[:loc]
  if dataset.startswith('zinc'):
    freq_file = ZINC_DATA_ROOT + 'freq.txt'
  elif dataset.startswith('cond'):
    freq_file = COND_DATA_ROOT + 'freq_final_max20.txt'
  else:
    freq_file = CHEMBL_DATA_ROOT + 'freq_final_' + dataset + '.txt'
  return read_molecule_mapping_from_freq_table(freq_file)

def read_bond_mapping_for_set(dataset):
  if dataset.startswith('zinc'):
    bond_map_file = ZINC_DATA_ROOT + 'freq.txt.bonds'
    bond_map = [([int(t) for t in line.split(': ')[0].split('_')])
                for line in read_strings_from_file(bond_map_file)]
  return bond_map

def normalize_graph(edges, node_types):
  edges = [e if e[0] <= e[1] else (e[1], e[0], e[2]) for e in edges]
  edges = sorted(edges, key=lambda p: (p[1], p[0]))
  edges = np.array(edges, dtype=np.int32).reshape(-1, 3)
  n_edges_for_node = np.bincount(edges[:, 1], minlength=len(node_types))
  node_types = np.array(node_types, dtype=np.int32)
  edge_types = edges[:, 2]
  edges = edges[:, :2]
  return edges, edge_types, node_types, n_edges_for_node

def permute_graph(edges, node_types, permutation=None):
  n_nodes = len(node_types)
  permute_idx = (np.random.permutation(n_nodes) if permutation is None
                   else permutation)
  inv_perm_idx = {p: i for i, p in enumerate(permute_idx)}
  node_types = [node_types[inv_perm_idx[i]] for i in range(n_nodes)]
  edges = [(permute_idx[e[0]], permute_idx[e[1]], e[2]) for e in edges]
  return edges, node_types

def get_molecule_from_file(f):
  try:
    _, n_bonds = [int(t) for t in f.readline().rstrip().split()]
    atoms = [int(a) for a in f.readline().rstrip().split()]
    bonds = []
    for _ in range(n_bonds):
      from_idx, to_idx, b_type = [int(b) for b in f.readline().rstrip().split()]
      bonds.append((from_idx, to_idx, b_type))
    return Molecule(atoms, bonds)
  except Exception:
    return None

def get_max_edge_type(molecule_list):
  return max(max(b[-1] for b in m.bonds) if len(m.bonds) > 0 else 0
             for m in molecule_list)

class Molecule(object):
  def __init__(self, atoms, bonds):
    self.atoms = atoms
    self.bonds = bonds

  def to_graph_str(self):
    graph_str = '%d %d\n' % (len(self.atoms), len(self.bonds))
    graph_str += ' '.join(['%d' % a for a in self.atoms])
    # pylint: disable=g-explicit-length-test
    graph_str += '' if len(self.bonds) == 0 else (
        '\n' + '\n'.join(['%d %d %d' % b for b in self.bonds]))
    return graph_str

