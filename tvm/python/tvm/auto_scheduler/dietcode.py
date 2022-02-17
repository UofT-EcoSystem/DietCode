import tvm

from tvm.runtime import Object
from . import _ffi_api


# <bojian/DietCode>
@tvm._ffi.register_object("auto_scheduler.DynWklDispatcher")
class DynWklDispatcher(Object):

    def dispatch(self, shape_tuple):
        sched, in_args = _ffi_api.DispatcherDispatch(
                             self, self._find_wkl_id(shape_tuple)
                         )
        return sched, in_args

    def dispatch_to_state(self, shape_tuple):
        return _ffi_api.DispatcherDispatchToState(
                   self, self._find_wkl_id(shape_tuple)
               )

    @property
    def states(self):
        return _ffi_api.DispatcherStates(self)

    @property
    def inst_disp_map(self):
        return _ffi_api.DispatcherInstDispMap(self)

    def _find_wkl_id(self, shape_tuple):
        from tvm.ir import Array

        if isinstance(shape_tuple, Array):
            shape_tuple = tuple([int(v) for v in list(shape_tuple)])
        for i, wkl_inst in enumerate(list(self.search_task.wkl_insts)):
            wkl_inst = tuple([int(v) for v in list(wkl_inst)])
            if wkl_inst == shape_tuple:
                return i
        assert False, "{} not found".format(shape_tuple)

    def get_skeleton(self, name):
        return _ffi_api.DispatcherGetSkeleton(self, name)

    def embed_compute_dag(self, compute_dag):
        return _ffi_api.DispatcherEmbedComputeDAG(self, compute_dag)


# def inline_dispatch(skeleton_mod_host, merged_mod_dev, dyn_wkl_dispatcher):
#     return _ffi_api.InlineDispatch(skeleton_mod_host, merged_mod_dev,
#                                    dyn_wkl_dispatcher)
def inline_dispatch(tensors, shape_vars, tree_classifier_root,
                    skeleton_mod_host, merged_mod_host, merged_mod_dev,
                    name
                    ):
    return _ffi_api.InlineDispatch(tensors, shape_vars,
                                   tree_classifier_root, 
                                   skeleton_mod_host, 
                                   merged_mod_host, merged_mod_dev,
                                   name)


def replace_shape_vars(wkl_func_args, shape_vars, new_shape_vars):
    replaced_dyn_args = \
            _ffi_api.ReplaceShapeVars(wkl_func_args, shape_vars, new_shape_vars)
    return tuple(replaced_dyn_args)

def instantiate_dyn_args(wkl_func_args, shape_vars, wkl_inst):
    instantiated_dyn_args = \
            _ffi_api.InstantiateDynArgs(wkl_func_args, shape_vars, wkl_inst)
    return tuple([i.value for i in instantiated_dyn_args])


@tvm._ffi.register_object("auto_scheduler.StateVer")
class StateVer(Object):
    def __init__(self, major, minor):
        self.__init_handle_by_constructor__(
                _ffi_api.StateVer,
                major, minor)


@tvm._ffi.register_object("auto_scheduler.DecisionTreeNode")
class DecisionTreeNode(Object):
    def __init__(self, predicate=None, if_node=None, else_node=None, 
                 state_ver=None):
        self.__init_handle_by_constructor__(
                _ffi_api.DecisionTreeNode,
                predicate,
                if_node,
                else_node,
                state_ver
                )
