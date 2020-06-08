#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# 
import struct 

def load_mnist(path, kind='train'):
    labels_path = path / f'{kind}-labels-idx1-ubyte'
    images_path = path / f'{kind}-images-idx3-ubyte'
    with labels_path.open('rb') as lbl:
        magic, n = struct.unpack('>II', lbl.read(8))
        labels = np.fromfile(lbl, dtype=np.uint8)
    with images_path.open('rb') as img:
        magic, num, rows, cols = struct.unpack('>IIII', img.read(16))
        images = np.fromfile(img, dtype=np.uint8)
        images = images.reshape(len(labels), rows*cols)
        images = ( (images/255.0) - 0.5 ) * 2.0
    return images, labels
