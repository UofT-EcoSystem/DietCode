import logging
import numpy as np
import pprint
import time

logger = logging.getLogger(__name__)

from ...shared.logger import AvgStdMedianLogger


class PySchedLogger:
    __slots__ = ['filename']

    def __init__(self, filename, append_mode):
        self.filename = filename
        if not append_mode:
            self._write_header()

    def _write_header(self):
        with open(self.filename, 'w') as fout:
            fout.write("""\
from tvm import te


""")

    def write_best_state(self, wkl_func_name, search_task, best_state):
        pysched = search_task.compute_dag.print_python_code_from_state(best_state)
        wkl_func_args = ', '.join([t.op.name for t in search_task.compute_dag.tensors])
        with open(self.filename, 'a') as fout:
            fout.write('def {}({}, s):\n'.format(wkl_func_name, wkl_func_args))
            fout.write('\n'.join(['    ' + line for line in pysched.split('\n')[:-1]]))
            fout.write('\n\n\n')

    def write_dyn_wkl_disp(self, sched_func_name_prefix, search_task,
                           dyn_wkl_dispatcher):
        wkl_func_args = ', '.join([t.op.name for t in search_task.compute_dag.tensors])
        best_states = list(dyn_wkl_dispatcher.states)
        inst_disp_map = dyn_wkl_dispatcher.inst_disp_map
        inst_disp_dict = dict()
        for k, v in inst_disp_map.items():
            shape_value = tuple(int(v) for v in search_task.wkl_insts[k.value])
            inst_disp_dict[shape_value] = v.value
        with open(self.filename, 'a') as fout:
            for state_id, best_state in enumerate(best_states):
                pysched = search_task.compute_dag.print_python_code_from_state(best_state)
                fout.write('def {}({}, s):\n'.format(sched_func_name_prefix + '_%d' % state_id,
                                                     wkl_func_args))
                fout.write('\n'.join(['    ' + line for line in pysched.split('\n')[:-1]]))
                fout.write('\n\n\n')
            fout.write('inst_disp_map = {}\n'.format(pprint.pformat(inst_disp_dict)))


class TFLOPSLogger:
    __slots__ = ['tflops_logger']

    def __init__(self, filename, append_mode):
        self.tflops_logger = AvgStdMedianLogger(filename, append_mode)

    def write(self, backend, attr, results, FLOPs):
        self.tflops_logger.write(
                backend, attr, 
                None if results is None else np.array(results))
