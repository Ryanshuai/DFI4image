#!/usr/bin/env python2


from interface import interface, method

class DeepFeatureRep(interface):
  r'''
  mean_F: \mathbb{R}^{N x H x W x 3} \to \mathbb{R}^{D}
  F_inverse: \mathbb{R}^{D} \to \mathbb{R}^{H x W x 3}

  mean_F achieves O(1) space wrt N if X is an iterator that yields
  \mathbb{R}^{H x W x 3}.
  '''
  mean_F=method(['self','X'])
  F_inverse=method(['self','F','initial_image'],keywords='options')
