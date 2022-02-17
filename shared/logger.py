import logging
import numpy as np
import os
import time

logger = logging.getLogger(__name__)


def remove_log_file(fname):
    try:
        os.remove(fname)
    except OSError:
        pass


class AvgStdMedianLogger:
    __slots__ = ['filename']

    def __init__(self, filename, append_mode):
        self.filename = filename
        if not append_mode:
            self._write_header()

    def _write_header(self):
        with open(self.filename, 'w') as fout:
            fout.write('Backend,ShapeTuple,Avg,STD,Median\n')
            
    def write(self, backend, shape_tuple, data=None):
        if data is None:
            with open(self.filename, 'a') as fout:
                fout.write('\"{}\",\"{}\",-,-,-\n'.format(backend, shape_tuple))
            logger.info("{} : (-)+(-) (M=(-))".format(backend))
        else:
            avg, std, median = np.average(data), np.std(data), np.median(data)
            with open(self.filename, 'a') as fout:
                fout.write('\"{}\",\"{}\",{},{},{}\n'
                               .format(backend, shape_tuple, avg, std, median))
            logger.info("{} : {}+{} (M={})".format(backend, avg, std, median))


class AutoSchedTimer:
    __slots__ = ['filename', 'attr', 'tic', 'toc']

    def __init__(self, filename, append_mode, attr):
        self.filename = filename
        if not append_mode:
            self._write_header()
        self.attr = attr

    def _write_header(self):
        with open(self.filename, 'w') as fout:
            fout.write("""\
"Task","Time Spent (s)"
""")

    def _write(self):
        with open(self.filename, 'a') as fout:
            fout.write('\"{}\",\"{}\"\n'.format(self.attr, self.toc - self.tic))
        logger.info("Auto-Scheduling Time for {} : {} s"
                        .format(self.attr, self.toc - self.tic))

    def __enter__(self):
        self.tic = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.time()
        self._write()
