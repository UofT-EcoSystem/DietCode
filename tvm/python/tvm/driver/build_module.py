# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""The build utils in python.
"""

from typing import Union, Optional, List, Mapping
import warnings

import tvm.tir

from tvm.runtime import Module
from tvm.runtime import ndarray
from tvm.ir import container
from tvm.ir import CallingConv
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.target import codegen
from tvm.te import tensor
from tvm.te import schedule
from tvm.target import Target
from tvm.tir.buffer import Buffer
from tvm.tir.expr import Var

from . import _ffi_api as ffi


def get_binds(args, compact=False, binds=None):
    """Internal function to get binds and arg_list given arguments.
    Parameters
    ----------
    args : list of Buffer or Tensor or Var
        The argument lists to the function.
    compact : bool
        If the statement has already bound to a compact buffer.
    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.
    Returns
    -------
    binds: dict
        The bind specification
    arg_list: list
        The list of symbolic buffers of arguments.
    """
    binds, arg_list = ffi.get_binds(args, compact, binds)
    return binds, arg_list


def schedule_to_module(
    sch: schedule.Schedule,
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
) -> IRModule:
    """According to the given schedule, form a function.
    Parameters
    ----------
    sch : tvm.te.schedule.Schedule
        The given scheduler to form the raw body
    args : list of Buffer or Tensor or Var
        The argument lists to the function.
    name : str
        The name of result function, default name is "main"
    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        The binds information
    Returns
    -------
    The body formed according to the given schedule
    """
    return ffi.schedule_to_module(sch, args, name, binds)


# <bojian/DietCode>
def _convert_decision_tree(tree, # feature_names
                           features):
    from sklearn.tree import _tree
    from ..auto_scheduler import StateVer, DecisionTreeNode

    tree_ = tree.tree_
    # print("features={}, tree.features={}".format(features, tree_.feature))
    # assert len(features) == len(tree_.feature), \
    #        "The number features must be consistent"
    # feature_name = [
    #     feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    #     for i in tree_.feature
    # ]
    features = [features[i] if i != _tree.TREE_UNDEFINED else None
                for i in tree_.feature
                ]
    # print("def tree({}):".format(", ".join(feature_names)))
    # CAUTION!: MUST store all the nodes otherwise they will be released
    tree_classifier_nodes = []

    def _recurse(node=0, depth=1):
        # indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # name = feature_name[node]
            var = features[node]
            threshold = int(tree_.threshold[node])
            # print("{}if {} <= {}:".format(indent, var, threshold))
            if_node = _recurse(tree_.children_left[node], depth + 1)
            # print("{}else:  # if {} > {}".format(indent, var, threshold))
            else_node = _recurse(tree_.children_right[node], depth + 1)
            tree_classifier_nodes.insert(
                    0, 
                    DecisionTreeNode(predicate=(var <= threshold),
                                     if_node=if_node, else_node=else_node))
            return tree_classifier_nodes[0]
        else:
            # print("{}return {}".format(indent, tree_.value[node]))
            # print(tree_.value[node])
            state_ver = StateVer(int(tree_.value[node][0][0]),
                                 int(tree_.value[node][1][0]))

            tree_classifier_nodes.insert(
                    0, DecisionTreeNode(state_ver=state_ver))
            return tree_classifier_nodes[0]

    _recurse()
    return tree_classifier_nodes


def _train_decision_tree(wkl_insts, wkl_inst_id_state_ver_map):
    from sklearn import tree
    tree_classifier = tree.DecisionTreeRegressor()

    X = []
    Y = []

    def _wkl_inst_id_to_list(wkl_inst_id):
        wkl_inst = wkl_insts[wkl_inst_id.value]
        return [i.value for i in list(wkl_inst)]

    for wkl_inst_id, state_ver in wkl_inst_id_state_ver_map.items():
        print("{} : {}".format(wkl_inst_id, state_ver))
        X.append(_wkl_inst_id_to_list(wkl_inst_id))
        Y.append([state_ver.major.value, state_ver.minor.value])

    # X = [[5, 768, 2304],
    #      [10, 768, 2304],
    #      [5, 768, 3072],
    #      [12, 768, 2304]
    #      ]
    # Y = [[0, 0], [0, 1], [1, 0], [2, 0]]

    print("X={}, Y={}".format(X, Y))
    tree_classifier.fit(X, Y)
    return tree_classifier


_check_no_opt_status = \
        tvm.tir.transform.Filter(
            lambda f: "already_opt" not in f.attrs or
                      f.attrs["already_opt"].value == False
                      
        )
_check_opt_status = \
        tvm.tir.transform.Filter(
            lambda f: "already_opt" in f.attrs and
                      f.attrs["already_opt"].value == True
        )

def _opt_mixed(input_mod, target):
    opt_mixed = [tvm.tir.transform.Apply(lambda f: f.with_attr("target", target)),
                 tvm.tir.transform.VerifyMemory(),
                 tvm.tir.transform.MergeDynamicSharedMemoryAllocations(),
                 ]
    if len(input_mod.functions) == 1:
        opt_mixed += [tvm.tir.transform.Apply(lambda f: f.with_attr("tir.is_entry_func", True))]
    if PassContext.current().config.get("tir.detect_global_barrier", False):
        opt_mixed += [tvm.tir.transform.ThreadSync("global")]
    opt_mixed += [tvm.tir.transform.ThreadSync("shared"),
                             tvm.tir.transform.ThreadSync("warp"),
                             tvm.tir.transform.InferFragment(),
                             tvm.tir.transform.LowerThreadAllreduce(),
                             tvm.tir.transform.MakePackedAPI(),
                             tvm.tir.transform.SplitHostDevice(),

                             tvm.tir.transform.MarkAlreadyOptFlag()
                             ]
    return tvm.transform.Sequential(opt_mixed)


_host_filter = \
        tvm.tir.transform.Filter(
            lambda f: "calling_conv" in f.attrs and \
                      f.attrs["calling_conv"].value != CallingConv.DEVICE_KERNEL_LAUNCH
        )

def _opt_host(target_host):
    opt_host = tvm.transform.Sequential([
                    _host_filter,

                    tvm.tir.transform.Apply(lambda f: f.with_attr("target", target_host)),
                    tvm.tir.transform.LowerTVMBuiltin(),
                    tvm.tir.transform.LowerDeviceStorageAccessInfo(),
                    tvm.tir.transform.LowerCustomDatatypes(),
                    tvm.tir.transform.LowerIntrin(),
                    tvm.tir.transform.CombineContextCall(),
                ])
    return opt_host


_dev_filter = \
        tvm.tir.transform.Filter(
            lambda f: "calling_conv" in f.attrs and \
                      f.attrs["calling_conv"].value == CallingConv.DEVICE_KERNEL_LAUNCH
        )

def _opt_dev():
    opt_device = tvm.transform.Sequential([
                 _dev_filter,
                 tvm.tir.transform.LowerWarpMemory(),
                 tvm.tir.transform.Simplify(),
                 tvm.tir.transform.LowerDeviceStorageAccessInfo(),
                 tvm.tir.transform.LowerCustomDatatypes(),
                 tvm.tir.transform.LowerIntrin(),
             ])
    return opt_device

def _merge_ir_mods(input_mods):
    ret = IRModule({})
    for mod in input_mods:
        ret.update(mod)
    return ret


# <bojian/DietCode>
@tvm._ffi.register_func("driver.lower_dyn_wkl_dispatcher")
def lower_dyn_wkl_dispatcher(
        dyn_wkl_dispatcher, target, target_host='llvm',
        name="default_function"
        # ret_rt_mod=True
        ):
    # print("dyn_wkl_dispatcher={}, target={}, target_host={}, name={}"
    #           .format(dyn_wkl_dispatcher, target, target_host, name)
    #       )
    # print("type(target)={}, type(target_host)={}, type(name)={}"
    #           .format(type(target), type(target_host), type(name))
    #       )

    # input_mods = []
    target, target_host = Target.check_and_update_host_consist(target, target_host)

    def _split_host_device(input_mod):
        input_mod = _opt_mixed(input_mod, target)(input_mod)
        return _host_filter(input_mod), _dev_filter(input_mod)


    # for i, wkl_inst in enumerate(dyn_wkl_dispatcher.search_task.wkl_insts):
    #     # print("wkl_inst={}".format(wkl_inst))
    #     sched, args = dyn_wkl_dispatcher.dispatch(wkl_inst)
    #     input_mod = lower(sched, args, name=name+('_%d'%i))
    #     input_mods.append(input_mod)
    #     # print("input_mod={}".format(input_mods[-1]))
    state_ver_ir_mod_map, wkl_inst_id_state_ver_map = \
            ffi.GenerateAndCompressIRMods(dyn_wkl_dispatcher, name)
    
    # print("state_ver_ir_mod_map={}, wkl_inst_id_state_ver_map={}"
    #           .format(state_ver_ir_mod_map, wkl_inst_id_state_ver_map))
    tree = _train_decision_tree(dyn_wkl_dispatcher.search_task.wkl_insts, wkl_inst_id_state_ver_map)
    
    shape_vars = dyn_wkl_dispatcher.search_task.shape_vars

    tree_classifier_nodes = _convert_decision_tree(tree, shape_vars)
    tree_classifier_root = tree_classifier_nodes[0]
    print(tree_classifier_root)

    input_mods = []
    for _, ir_mod in state_ver_ir_mod_map.items():
        input_mods.append(ir_mod)

    merged_mod = _merge_ir_mods(# dyn_wkl_dispatcher, input_mods
                                input_mods
                                # state_ver_ir_mod_map
                                )
    merged_mod_host, merged_mod_dev = _split_host_device(merged_mod)

    skeleton_mod = dyn_wkl_dispatcher.get_skeleton(name)
    skeleton_mod_host, _ = _split_host_device(skeleton_mod)

    # print("skeleton_mod_host={}".format(skeleton_mod_host))

    # assert merged_rt_mod_dev, "Device module is not defined"
    # print("skeleton_mod_host={}, merged_mod_dev={}"
    #           .format(skeleton_mod_host, merged_mod_dev))
    # print("skeleton_mod_host={}".format(skeleton_mod_host))
    from tvm import auto_scheduler

    tensor_args = dyn_wkl_dispatcher.search_task.compute_dag.tensors

    skeleton_mod_host = auto_scheduler.inline_dispatch(
                            tensor_args, shape_vars,
                            tree_classifier_root,
                            skeleton_mod_host, merged_mod_host, merged_mod_dev,
                            name)
    print("skeleton_mod_host={}".format(skeleton_mod_host))

    # if ret_rt_mod:
    #     skeleton_rt_mod_host = codegen.build_module(skeleton_mod_host, target_host)
    #     skeleton_rt_mod_host.import_module(merged_rt_mod_dev)
    #     return OperatorModule.from_module(
    #                skeleton_rt_mod_host,
    #                ir_module_by_target={target: merged_mod,
    #                                     target_host: skeleton_mod_host
    #                                     },
    #                name=name)
    # else:
    #     return [skeleton_mod_host, merged_mod_dev]
    # mod_host, mod_dev, _ = \
    #         _build_for_device(_merge_ir_mods([skeleton_mod_host, merged_mod_dev]),
    #                           target, target_host)
    return [# {str(target): mod_dev, str(target_host) : mod_host},
            _merge_ir_mods([_opt_host(target_host)(skeleton_mod_host),
                            _opt_dev()(merged_mod_dev)]),
            list(tensor_args) + list(shape_vars)]
    # return skeleton_mod_host, merged_mod_dev


# <bojian/DietCode>
# def dyn_build
# @tvm._ffi.register_func("driver.build_dyn_func")
# def build_dyn_func(dyn_wkl_dispatcher, target, target_host='llvm'):
#     skeleton_mod_host, merged_mod_dev = \
#             tvm.lower_dyn_wkl_dispatcher(dyn_wkl_dispatcher, target=target)

#     rt_mod_host = tvm.target.codegen.build_module(skeleton_mod_host,
#                                                   target=target_host)
#     rt_mod_dev  = tvm.target.codegen.build_module(merged_mod_dev, target=target)
#     rt_mod_host.import_module(rt_mod_dev)
#     return rt_mod_host


def lower(
    inp: Union[schedule.Schedule, PrimFunc, IRModule],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    name: str = "main",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
    simple_mode: bool = False,
) -> IRModule:
    """Lowering step before build into target.

    Parameters
    ----------
    inp : Union[tvm.te.schedule.Schedule, tvm.tir.PrimFunc, IRModule]
        The TE schedule or TensorIR PrimFunc/IRModule to be built

    args : Optional[List[Union[tvm.tir.Buffer, tensor.Tensor, Var]]]
        The argument lists to the function for TE schedule.
        It should be None if we want to lower TensorIR.

    name : str
        The name of the result function.

    binds : Optional[Mapping[tensor.Tensor, tvm.tir.Buffer]]
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    simple_mode : bool
        Whether only output simple and compact statement, this will skip
        LoopPartition, api wrapper generation and Unrolling.

    Returns
    -------
    m : IRModule
       The result IRModule
    """
    if isinstance(inp, IRModule):
        return ffi.lower_module(inp, simple_mode)
    if isinstance(inp, PrimFunc):
        return ffi.lower_primfunc(inp, name, simple_mode)
    if isinstance(inp, schedule.Schedule):
        return ffi.lower_schedule(inp, args, name, binds, simple_mode)
    raise ValueError("Expected input to be an IRModule, PrimFunc or Schedule, but got, ", type(inp))


def _build_for_device(input_mod, target, target_host):
    """Build the lowered functions for a device with the given compilation
    target.

    Parameters
    ----------
    input_mod : IRModule
        The schedule to be built.

    target : str or :any:`tvm.target.Target`
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target`
        The host compilation target.

    Returns
    -------
    fhost : IRModule
        The host IRModule.

    mdev : tvm.module
        A module that contains device code.
    """
    target, target_host = Target.check_and_update_host_consist(target, target_host)
    device_type = ndarray.device(target.kind.name, 0).device_type


    # <bojian/DietCode>
    # print("input_mod={}".format(input_mod))
    # optimized_input_mod = check_opt_status(input_mod)
    # optimized_mod_host, optimized_mod_dev = None, None
    # if len(optimized_input_mod.functions) != 0:
    #     optimized_mod_host, optimized_mod_dev = host_filter(input_mod), dev_filter(input_mod)
    # input_mod = check_no_opt_status(input_mod)

    # <bojian/DietCode>
    # mod_mixed = input_mod
    opt_mod_mixed = _check_opt_status(input_mod)
    no_opt_mod_mixed = _check_no_opt_status(input_mod)
    # mod_mixed = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod_mixed)

    # opt_mixed = [

    #     # <bojian/DietCode>
    #     check_no_opt_status,

    #     tvm.tir.transform.VerifyMemory(),
    #     tvm.tir.transform.MergeDynamicSharedMemoryAllocations(),
    # ]
    # if len(mod_mixed.functions) == 1:
    #     opt_mixed += [tvm.tir.transform.Apply(lambda f: f.with_attr("tir.is_entry_func", True))]

    # if PassContext.current().config.get("tir.detect_global_barrier", False):
    #     opt_mixed += [tvm.tir.transform.ThreadSync("global")]
    # opt_mixed += [
    #     tvm.tir.transform.ThreadSync("shared"),
    #     tvm.tir.transform.ThreadSync("warp"),
    #     tvm.tir.transform.InferFragment(),
    #     tvm.tir.transform.LowerThreadAllreduce(),
    #     tvm.tir.transform.MakePackedAPI(),
    #     tvm.tir.transform.SplitHostDevice(),
    # ]
    # <bojian/DietCode>
    # mod_mixed = tvm.transform.Sequential(opt_mixed)(mod_mixed)
    # no_opt_mod_mixed = tvm.transform.Sequential(opt_mixed)(mod_mixed)
    no_opt_mod_mixed = _opt_mixed(input_mod, target)(no_opt_mod_mixed)

    # <bojian/DietCode>
    # opt_mod_mixed = check_opt_status(mod_mixed)
    # if len(opt_mod_mixed.functions) != 0:
    #     no_opt_mod_mixed.update(opt_mod_mixed)
    # no_opt_mod_mixed.update(opt_mod_mixed)
    # mod_mixed = no_opt_mod_mixed
    # print("mod_mixed={}".format(mod_mixed))


    # <bojian/DietCode>
    # device optimizations
    # opt_device = tvm.transform.Sequential(
    #     [
    #         # <bojian/DietCode>
    #         # tvm.tir.transform.Filter(
    #         #     lambda f: "calling_conv" in f.attrs
    #         #     and f.attrs["calling_conv"].value == CallingConv.DEVICE_KERNEL_LAUNCH
    #         # ),
    #         dev_filter,

    #         tvm.tir.transform.LowerWarpMemory(),
    #         tvm.tir.transform.Simplify(),
    #         tvm.tir.transform.LowerDeviceStorageAccessInfo(),
    #         tvm.tir.transform.LowerCustomDatatypes(),
    #         tvm.tir.transform.LowerIntrin(),

    #         # <bojian/DietCode>
    #         # tvm.tir.transform.MarkAlreadyOptFlag()

    #     ]
    # )
    # mod_dev = opt_device(mod_mixed)
    mod_dev = _opt_dev()(no_opt_mod_mixed)
    opt_mod_dev = _dev_filter(opt_mod_mixed)
    mod_dev.update(opt_mod_dev)


    # host optimizations
    # opt_host = tvm.transform.Sequential(
    #     [
    #         # <bojian/DietCode>
    #         # tvm.tir.transform.Filter(
    #         #     lambda f: "calling_conv" not in f.attrs
    #         #     or f.attrs["calling_conv"].value != CallingConv.DEVICE_KERNEL_LAUNCH
    #         # ),
    #         host_filter,

    #         tvm.tir.transform.Apply(lambda f: f.with_attr("target", target_host)),
    #         tvm.tir.transform.LowerTVMBuiltin(),
    #         tvm.tir.transform.LowerDeviceStorageAccessInfo(),
    #         tvm.tir.transform.LowerCustomDatatypes(),
    #         tvm.tir.transform.LowerIntrin(),
    #         tvm.tir.transform.CombineContextCall(),

    #         # <bojian/DietCode>
    #         # tvm.tir.transform.MarkAlreadyOptFlag()

    #     ]
    # )
    # mod_host = opt_host(mod_mixed)
    mod_host = _opt_host(target_host)(no_opt_mod_mixed)
    opt_mod_host = _host_filter(opt_mod_mixed)
    mod_host.update(opt_mod_host)

    # <bojian/DietCode>
    # if optimized_mod_host is not None:
    #     mod_host.update(optimized_mod_host)
    # print("mod_dev={}, mod_host={}".format(mod_dev, mod_host))


    if device_type == ndarray.cpu(0).device_type and target_host == target:
        assert len(mod_dev.functions) == 0
    if "gpu" in target.keys and len(mod_dev.functions) == 0:
        warnings.warn(
            "Specified target %s, but cannot find device code, did you do " "bind?" % target
        )

    rt_mod_dev = codegen.build_module(mod_dev, target) if len(mod_dev.functions) != 0 else None

    # <bojian/DietCode>
    # return mod_host, rt_mod_dev
    return mod_host, mod_dev, rt_mod_dev


def build(
    inputs: Union[schedule.Schedule, PrimFunc, IRModule, Mapping[str, IRModule]],
    args: Optional[List[Union[Buffer, tensor.Tensor, Var]]] = None,
    target: Optional[Union[str, Target]] = None,
    target_host: Optional[Union[str, Target]] = None,
    name: Optional[str] = "default_function",
    binds: Optional[Mapping[tensor.Tensor, Buffer]] = None,
):
    """Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.

    Parameters
    ----------
    inputs : Union[tvm.te.schedule.Schedule, tvm.tir.PrimFunc, IRModule, Mapping[str, IRModule]]
        The input to be built

    args : Optional[List[Union[tvm.tir.Buffer, tensor.Tensor, Var]]]
        The argument lists to the function.

    target : Optional[Union[str, Target]]
        The target and option of the compilation.

    target_host : Optional[Union[str, Target]]
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    name : Optional[str]
        The name of result function.

    binds : Optional[Mapping[tensor.Tensor, tvm.tir.Buffer]]
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    Returns
    -------
    ret : tvm.module
        A module that combines both host and device code.

    Examples
    ________
    There are two typical example uses of this function depending on the type
    of the argument `inputs`:
    1. it is an IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s = tvm.te.create_schedule(C.op)
        m = tvm.lower(s, [A, B, C], name="test_add")
        rt_mod = tvm.build(m, target="llvm")

    2. it is a dict of compilation target to IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s1 = tvm.te.create_schedule(C.op)
        with tvm.target.cuda() as cuda_tgt:
          s2 = topi.cuda.schedule_injective(cuda_tgt, [C])
          m1 = tvm.lower(s1, [A, B, C], name="test_add1")
          m2 = tvm.lower(s2, [A, B, C], name="test_add2")
          rt_mod = tvm.build({"llvm": m1, "cuda": m2}, target_host="llvm")

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    """
    if isinstance(inputs, schedule.Schedule):
        if args is None:
            raise ValueError("args must be given for build from schedule")
        input_mod = lower(inputs, args, name=name, binds=binds)
    elif isinstance(inputs, (list, tuple, container.Array)):
        merged_mod = tvm.IRModule({})
        for x in inputs:
            merged_mod.update(lower(x))
        input_mod = merged_mod
    elif isinstance(inputs, (tvm.IRModule, PrimFunc)):
        # <bojian/DietCode>
        # cannot lower the module again, otherwise some functions arguments
        # might be seen as not declared
        input_mod = lower(inputs)
        # pass
        # input_mod = inputs
    elif not isinstance(inputs, (dict, container.Map)):
        raise ValueError(
            f"Inputs must be Schedule, IRModule or dict of target to IRModule, "
            f"but got {type(inputs)}."
        )

    if not isinstance(inputs, (dict, container.Map)):
        target = Target.current() if target is None else target
        target = target if target else "llvm"
        target_input_mod = {target: input_mod}
    else:
        target_input_mod = inputs

    for tar, mod in target_input_mod.items():
        if not isinstance(tar, (str, Target)):
            raise ValueError("The key of inputs must be str or " "Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be Schedule, IRModule," "or dict of str to IRModule.")

    target_input_mod, target_host = Target.check_and_update_host_consist(
        target_input_mod, target_host
    )


    # <bojian/DietCode>
    # print("target_input_mod={}".format(target_input_mod, target_host))


    if not target_host:
        for tar, mod in target_input_mod.items():
            tar = Target(tar)
            device_type = ndarray.device(tar.kind.name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"

    target_input_mod, target_host = Target.check_and_update_host_consist(
        target_input_mod, target_host
    )


    # <bojian/DietCode>
    # print("target_host={}".format(target_host))


    mod_host_all = tvm.IRModule({})

    device_modules = []
    for tar, input_mod in target_input_mod.items():
        mod_host, _, mdev = _build_for_device(input_mod, tar, target_host)
        mod_host_all.update(mod_host)
        device_modules.append(mdev)

    # Generate a unified host module.
    rt_mod_host = codegen.build_module(mod_host_all, target_host)


    # <bojian/DietCode>
    # print("mod_host_all={}, rt_mod_host={}".format(mod_host_all, rt_mod_host))


    # Import all modules.
    for mdev in device_modules:
        if mdev:
            rt_mod_host.import_module(mdev)

    if not isinstance(target_host, Target):
        target_host = Target(target_host)
    if (
        target_host.attrs.get("runtime", tvm.runtime.String("c++")) == "c"
        and target_host.attrs.get("system-lib", 0) == 1
    ):
        if target_host.kind.name == "c":
            create_csource_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateCSourceCrtMetadataModule"
            )
            to_return = create_csource_crt_metadata_module([rt_mod_host], target_host)

        elif target_host.kind.name == "llvm":
            create_llvm_crt_metadata_module = tvm._ffi.get_global_func(
                "runtime.CreateLLVMCrtMetadataModule"
            )
            to_return = create_llvm_crt_metadata_module([rt_mod_host], target_host)
    else:
        to_return = rt_mod_host

    return OperatorModule.from_module(to_return, ir_module_by_target=target_input_mod, name=name)


class OperatorModule(Module):
    """Wraps the Module returned by tvm.build() and captures additional outputs of that function."""

    @classmethod
    def from_module(cls, mod, **kwargs):
        # NOTE(areusch): It is generally unsafe to continue using `mod` from this point forward.
        # If an exception occurs in cls.__init__, handle will be deleted. For this reason,
        # set mod.handle to None.
        handle = mod.handle
        mod.handle = None
        return cls(handle, **kwargs)

    def __init__(self, handle, ir_module_by_target=None, name=None):
        super(OperatorModule, self).__init__(handle)
        self.ir_module_by_target = ir_module_by_target
        self.name = name
