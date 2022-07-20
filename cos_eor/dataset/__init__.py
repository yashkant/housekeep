#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from habitat.core.dataset import Dataset
# from habitat.core.registry import registry


def _try_register_cos_eor_dataset():
    from cos_eor.dataset.dataset import CosRearrangementDatasetV0
