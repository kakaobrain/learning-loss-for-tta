import os
import pickle

import json

import logging
import torchvision
from tqdm._tqdm import tqdm

from commons import get_logger
from conf import dataroot

_logger = get_logger('CachedDataLoader', logging.WARNING)


class CachedDataLoader(object):
    NO_SEPARATE_EPOCH = 9999991958
    CACHE_STATUS_UNKNOWN = 0
    CACHE_STATUS_NO = -1
    CACHE_STATUS_EXIST = 2

    def __init__(self, loader, tag='cache', cache_max=5, step_max=10e10):
        self.epoch_cnt = -1
        self.i = 0
        self.step_max = step_max

        self.cache_max = cache_max
        self.loader = loader
        self.loader_iter = None
        self.tag = tag
        self.cache_status = CachedDataLoader.CACHE_STATUS_UNKNOWN

    def __iter__(self):
        if self.cache_max != CachedDataLoader.NO_SEPARATE_EPOCH:
            self.epoch_cnt += 1
        else:
            self.epoch_cnt = 0
        self.i = -1
        if self.loader_iter is not None:
            # self.loader_iter.close()
            # del self.loader_iter
            pass
        self.loader_iter = iter(self.loader)
        _logger.debug('cache epoch %d' % self.epoch_cnt)
        return self

    def __next__(self):
        self.i += 1
        if self.i > len(self):
            raise StopIteration
        filepath = self._path(self.i)
        item = None
        if os.path.exists(filepath) and self.epoch_cnt < self.cache_max:
            item = self._load(self.i)
            if self.cache_status != CachedDataLoader.CACHE_STATUS_EXIST:
                self.cache_status = CachedDataLoader.CACHE_STATUS_EXIST
                _logger.warning(f'DataCache Exist. epoch={self.epoch_cnt}')
        if item is not None:
            return item
        item = next(self.loader_iter)
        if self.cache_status != CachedDataLoader.CACHE_STATUS_NO:
            self.cache_status = CachedDataLoader.CACHE_STATUS_NO
            _logger.warning(f'DataCache Non-Exist. epoch={self.epoch_cnt}')
        if self.epoch_cnt < self.cache_max:
            self._save(self.i, item)
        return item

    def _path(self, i):
        filepath = f'_cachedl/{self.tag}_e{self.epoch_cnt:04}_{i:08}.pickle'
        return filepath

    def _save(self, i, item):
        filepath = self._path(i)
        with open(filepath, 'wb') as f:
            pickle.dump(item, f)

    def _load(self, i):
        try:
            filepath = self._path(i)
            with open(filepath, 'rb') as f:
                item = pickle.load(f)
            return item
        except Exception as e:
            print(e)
            return None

    def __len__(self):
        return min(len(self.loader), self.step_max)
