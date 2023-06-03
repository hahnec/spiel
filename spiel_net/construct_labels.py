import numpy
import torch


def construct_labels_indexing(toas, gt_samples):

    lib = numpy if isinstance(toas, numpy.ndarray) else torch
    
    toa_diff = abs(toas - gt_samples[:, None])

    # create inverted labels (for contrastive loss zero corresponds to target)
    comp_idcs = lib.argmin(toa_diff, axis=1)
    labels = lib.ones_like(toas)
    labels[lib.arange(labels.shape[0]), comp_idcs] = 0

    return labels, comp_idcs