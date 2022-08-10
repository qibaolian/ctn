# -*- coding=utf-8 -*-
"""
    Author: wanghao
    date: 2019/10/26
"""

import torch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        # for compat python v2.*.*
        # self.next = self.__next__
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preload(self):
        data = next(self.loader)  # may rasie StopIteration
        self.next_input, self.next_target = data['img'], data['gt'].long()
        # self.next_input, self.next_target = data['img'].to(self.device), data['gt'].long().to(self.device)
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preload()
        input = self.next_input
        target = self.next_target
        return input, target

    def __len__(self):
        return len(self.loader)
