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
        if filename is not None and not append_mode:
            self._write_header()

    def _write_header(self):
        with open(self.filename, 'w') as fout:
            fout.write('Backend,ShapeTuple,Avg,STD,Median\n')
            
    def write(self, backend, shape_tuple, data=None):
        if data is None:
            if self.filename is not None:
                with open(self.filename, 'a') as fout:
                    fout.write(f'\"{backend}\",\"{shape_tuple}\",-,-,-\n')
            logger.info(f"{backend} : (-)+(-) (M=(-))")
        else:
            avg, std, median = np.average(data), np.std(data), np.median(data)
            if self.filename is not None:
                with open(self.filename, 'a') as fout:
                    fout.write(f'\"{backend}\",\"{shape_tuple}\",{avg},{std},{median}\n')
            logger.info(f"{backend} @{shape_tuple} : {avg}+{std} (M={median})")


class ScopedTimer:
    __slots__ = ['filename', 'name', 'tic', 'toc']

    def __init__(self, filename, append_mode, name):
        self.filename = filename
        if filename is not None and not append_mode:
            self._write_header()
        self.name = name

    def _write_header(self):
        with open(self.filename, 'w') as fout:
            fout.write("""\
"Task","Time Spent (s)"
""")

    def _write(self):
        if self.filename is not None:
            with open(self.filename, 'a') as fout:
                fout.write(f'\"{self.name}\",\"{self.toc - self.tic}\"\n')
        logger.info(f"Time for {self.name} : {self.toc - self.tic} s")

    def __enter__(self):
        self.tic = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.time()
        self._write()
