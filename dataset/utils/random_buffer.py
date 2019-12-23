from collections import deque
from typing import Callable, Type, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as dist


class RandomBuffer:
    """
    RandomBuffer recycles elements which are already loaded in memory based on usage frequency.
    More hits element has more probable swap for new element is. So elements in memory create
    kind of a batching pool which should create original batches with high probability even though
    elements are reused.
    """
    def __init__(self, buffer_size, swap_prob_offset, show_stats=False):
        self.show_stats = show_stats
        self.swap_prob_offset = swap_prob_offset
        assert -1.0 <= self.swap_prob_offset
        assert self.swap_prob_offset <= 1.0

        self._buffer = [None]*buffer_size
        self._idx_sampler = dist.Uniform(0, len(self._buffer))

        self._swap_hist = deque(maxlen=10000)
        self.getitem_counter = 0

    def fill(self, new_element_generator: Iterator[Type]):
        """
        Initialize (prefill) buffer with elements (must be done if show_stats=True).

        Example:
        self.buffer.fill((self._new_img_buf(idx) for idx in range(len(self))))

        :param new_element_generator: Generator with enough elements to fill buffer.
        """
        for i, element in enumerate(new_element_generator):
            if i == len(self._buffer):
                return
            self._buffer[i] = element
            self._buffer[i].hits = 0
            self._swap_hist.append(0)

    def __getitem__(self, new_element_method: Callable[[], Type]):
        """
        Returns either random element from buffer or uses new_element_method to create new and swaps them.
        :param new_element_method: Returns new buffer element (it is kind of lazy on demand load).
        :return: Buffer element.
        """
        self.getitem_counter += 1
        # select random element from buffer
        rnd_idx = int(self._idx_sampler.sample())
        # initialize if needed (self.fill method was not used)
        if self._buffer[rnd_idx] is None:
            self._buffer[rnd_idx] = new_element_method()
            self._buffer[rnd_idx].hits = 0
        img_buf = self._buffer[rnd_idx]
        # calculate swap probability based on number of hits
        swap_prob = self.swap_prob_offset + 1 - (1 / (1 + img_buf.hits))
        swap_prob = min(max(0.0, swap_prob), 1.0)
        self._swap_hist.append(0)
        # flip unbalanced coin and with P(1)=swap_prob swap element
        if bool(dist.Binomial(1, swap_prob).sample()):
            img_buf = new_element_method()
            self._buffer[rnd_idx] = img_buf
            self._buffer[rnd_idx].hits = 0
            self._swap_hist[-1] = 1
        img_buf.hits += 1
        # periodically create pyplot histograms with swap history and element hits
        if self.show_stats:
            self.stats()
        return img_buf

    def stats(self):
        """
        Periodically create pyplot histograms with swap history and element hits.
        """
        if self.getitem_counter % 1000 == 0:
            swaps = np.sum(self._swap_hist)
            print('swaps={}, swaps/get_cntr={}'.format(swaps, swaps / self.getitem_counter))
            conv_window_size = 100
            plt.hist(np.convolve(self._swap_hist, [1] * conv_window_size, 'valid'))
            plt.title('Swap history (convolved window={})'.format(conv_window_size))
            plt.xlabel('Swaps')
            plt.ylabel('Windows')
            plt.show()
            plt.hist([b.hits for b in self._buffer])
            plt.title('Element hits')
            plt.xlabel('Hits')
            plt.ylabel('Elements')
            plt.show()


if __name__ == '__main__':
    class TestElement:
        def __init__(self, idx):
            self.idx = idx
    rndbuf = RandomBuffer(100, -0.4, True)
    rndbuf.fill((TestElement(idx) for idx in range(100)))
    for i in range(10000):
        element = rndbuf[lambda: TestElement(i)]
        print('Elem idx', element.idx)
