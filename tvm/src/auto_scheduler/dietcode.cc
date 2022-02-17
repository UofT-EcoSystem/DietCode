#include <tvm/auto_scheduler/dietcode.h>

// <bojian/DietCode>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/tir/transform.h>

#include "./utils.h"


namespace tvm {
namespace auto_scheduler {


TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DynWklDispatcherNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DynWklDispatcherNode*>(ref.get());
      p->stream << "DynWklDispatcher("
                << node->search_task << ", "
                << ArrayToString(node->states) << ", "
                << MapToString(node->inst_disp_map) << ")";
    });


DynWklDispatcher::DynWklDispatcher(
    const SearchTask& search_task, std::vector<State>&& states,
    std::unordered_map<size_t, size_t>&& inst_disp_map) {
  ObjectPtr<DynWklDispatcherNode> node = make_object<DynWklDispatcherNode>();
  node->search_task = search_task;
  node->states = std::move(states);
  node->inst_disp_map = std::move(inst_disp_map);
  data_ = std::move(node);
}

Array<ObjectRef>
DynWklDispatcherNode::Dispatch(const int wkl_id) const {
  const State& state = states[inst_disp_map.at(wkl_id)];
  // LOG(INFO) << "wkl_inst=" << ArrayToString(search_task->wkl_insts[wkl_id]);
  std::pair<te::Schedule, Array<te::Tensor>> sch_and_tensors =
      search_task->compute_dag.InstantiateAndApplySteps(
        state, search_task->shape_vars.value(),
        ToPrimExprArray(search_task->wkl_insts[wkl_id]));
  return {sch_and_tensors.first, sch_and_tensors.second};
}

State DynWklDispatcherNode::DispatchToState(const int wkl_id) const {
  return states[inst_disp_map.at(wkl_id)];
}

void
DynWklDispatcherNode::EmbedComputeDAG(const ComputeDAG& compute_dag) {
  SearchTaskNode* const mutable_search_task = search_task.CopyOnWrite();
  mutable_search_task->compute_dag = compute_dag;
}

// Array<ObjectRef>
IRModule
DynWklDispatcherNode::GetSkeleton(const String& name) const {
  // Array<PrimExpr> shape_values;

  std::pair<te::Schedule, Array<te::Tensor>> sch_and_tensors =
      search_task->compute_dag.InstantiateAndApplySteps(
        states[0], search_task->shape_vars.value(),
        ToPrimExprArray(search_task->shape_vars.value()));
  Array<ObjectRef> args;
  for (const te::Tensor& t : sch_and_tensors.second) {
    args.push_back(t);
  }
  for (const DynShapeVar& shape_var : search_task->shape_vars.value()) {
    args.push_back(shape_var);
  }
  // return {sch_and_tensors.first, sch_and_tensors.second};
  return LowerSchedule(sch_and_tensors.first, args, name, {});
}


static inline size_t BitVectorToInt(const std::vector<bool>& bit_vector) {
  size_t ret = 0;
  for (const bool b : bit_vector) {
    ret = ret * 2 + static_cast<size_t>(b);
  }
  return ret;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StateVerNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const StateVerNode*>(ref.get());
      p->stream << "StateVer(" << node->major << ", " << node->minor << ")";
    });

Array<ObjectRef>
DynWklDispatcherNode::GenerateAndCompressIRMods(const String& prefix) const {
  std::unordered_map<size_t, std::unordered_set<size_t>> inst_disp_reverse_map;
  std::vector<std::vector<bool>> wkl_insts_oob_marker;
  wkl_insts_oob_marker.reserve(search_task->wkl_insts.size());

  for (size_t i = 0; i < search_task->wkl_insts.size();
       ++i) {
    size_t state_id = inst_disp_map.at(i);
    inst_disp_reverse_map[state_id].insert(i);

    Array<IntImm> wkl_inst = search_task->wkl_insts[i];
    State state = states[state_id];
    state = this->search_task->compute_dag.InferBound(state);
    wkl_insts_oob_marker.push_back(
        state.GetOOBMarkerOnWklInst(
          search_task->shape_vars.value(), wkl_inst
        ));
  }
  Map<Integer, StateVer> new_inst_disp_map;
  std::unordered_map<StateVer, std::unordered_set<size_t>, StructuralHash, StructuralEqual>
      new_inst_disp_reverse_map;

  for (const auto& state_id_to_wkl_inst_ids : inst_disp_reverse_map) {
    std::unordered_map<size_t, size_t> state_minor_counter;

    for (const size_t wkl_inst_id : state_id_to_wkl_inst_ids.second) {
      size_t oob_marker = BitVectorToInt(wkl_insts_oob_marker[wkl_inst_id]);

      if (!state_minor_counter.count(oob_marker)) {
        state_minor_counter.emplace(oob_marker, state_minor_counter.size());
      }
      StateVer state_ver = 
          StateVer(state_id_to_wkl_inst_ids.first, state_minor_counter[oob_marker]);

      new_inst_disp_map.Set(Integer(wkl_inst_id), state_ver);
      new_inst_disp_reverse_map[state_ver].insert(wkl_inst_id);
    }
  }

  Map<StateVer, IRModule> compressed_ir_mods;

  for (const auto& state_ver_to_wkl_inst_ids : new_inst_disp_reverse_map) {
    const StateVer& state_ver = state_ver_to_wkl_inst_ids.first;
    Array<Array<IntImm>> wkl_insts;
    for (const size_t wkl_inst_id : state_ver_to_wkl_inst_ids.second) {
      wkl_insts.push_back(search_task->wkl_insts[wkl_inst_id]);
    }

    std::pair<te::Schedule, Array<te::Tensor>> sch_and_tensors =
        search_task->compute_dag.InstantiateAndApplySteps(
          states[state_ver->major], search_task->shape_vars.value(),
          ToPrimExprArray(search_task->shape_vars.value())
        );
    Array<ObjectRef> args;
    for (const te::Tensor& t : sch_and_tensors.second) {
      args.push_back(t);
    }
    for (const DynShapeVar& dyn_shape_var : search_task->shape_vars.value()) {
      args.push_back(dyn_shape_var);
    }
    IRModule ir_mod = LowerSchedule(sch_and_tensors.first, args,
                                    std::string(prefix)
                                      + "_" + std::to_string(state_ver->major->value)
                                      + "_" + std::to_string(state_ver->minor->value),
                                    {},
                                    false,
                                    search_task->shape_vars.value(),
                                    wkl_insts
                                    
                                    );
    // LOG(FATAL) << "ir_mod=" << ir_mod;
    compressed_ir_mods.Set(state_ver, ir_mod);
  }
  return {compressed_ir_mods, new_inst_disp_map};
}


TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherDispatch")
    .set_body_typed([](const DynWklDispatcher& dispatcher, const int wkl_id) {
      return dispatcher->Dispatch(wkl_id);
    });


TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherDispatchToState")
    .set_body_typed([](const DynWklDispatcher& dispatcher, const int wkl_id) {
      return dispatcher->DispatchToState(wkl_id);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherGetSkeleton")
    .set_body_typed([](const DynWklDispatcher& dispatcher,
                       const String& name) {
      return dispatcher->GetSkeleton(name);
    });


TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherStates")
    .set_body_typed([](const DynWklDispatcher& dispatcher) {
      Array<State> states;
      for (const State& state : dispatcher->states) {
        states.push_back(state);
      }
      return states;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherInstDispMap")
    .set_body_typed([](const DynWklDispatcher& dispatcher) {
      Map<Integer, Integer> inst_disp_map;
      for (const std::pair<size_t, size_t>& kv_pair :
           dispatcher->inst_disp_map) {
        inst_disp_map.Set(kv_pair.first, kv_pair.second);
      }
      return inst_disp_map;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherEmbedComputeDAG")
    .set_body_typed([](DynWklDispatcher dispatcher,
                       const ComputeDAG& compute_dag) {
      DynWklDispatcherNode* const mutable_dispatcher = dispatcher.CopyOnWrite();
      mutable_dispatcher->EmbedComputeDAG(compute_dag);
      return dispatcher;
    });

TVM_REGISTER_NODE_TYPE(DynWklDispatcherNode);

StateVer::StateVer(const int major, const int minor) {
  ObjectPtr<StateVerNode> node = make_object<StateVerNode>();
  node->major = Integer(major);
  node->minor = Integer(minor);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("auto_scheduler.StateVer")
    .set_body_typed([](const int major, const int minor) {
      return StateVer(major, minor);
    });

TVM_REGISTER_NODE_TYPE(StateVerNode);


DecisionTreeNode::DecisionTreeNode(const StateVer& state_ver) {
  ObjectPtr<DecisionTreeNodeNode> node = make_object<DecisionTreeNodeNode>();
  node->state_ver = state_ver;
  data_ = std::move(node);
}

DecisionTreeNode::DecisionTreeNode(const PrimExpr& predicate,
                                   const DecisionTreeNodeNode* const if_node,
                                   const DecisionTreeNodeNode* const else_node) {
  ObjectPtr<DecisionTreeNodeNode> node = make_object<DecisionTreeNodeNode>();
  node->predicate = predicate;
  node->if_node = if_node;
  node->else_node = else_node;
  data_ = std::move(node);
}

static void printDecisionTreeRecursively(
    std::ostream& out,
    const DecisionTreeNodeNode* const tree_node,
    const std::string& indent = "  ") {
  CHECK(tree_node != nullptr);
  if (tree_node->predicate) {
    CHECK(tree_node->predicate.value().defined());
    CHECK(tree_node->if_node != nullptr);
    CHECK(tree_node->else_node != nullptr);
    out << indent << "if (" << tree_node->predicate << ") {" "\n";
    printDecisionTreeRecursively(out, tree_node->if_node, indent + "  ");
    out << indent << "} else {" "\n";
    printDecisionTreeRecursively(out, tree_node->else_node, indent + "  ");
    out << indent << "}" "\n";
  } else {

    CHECK(tree_node->state_ver);
    out << indent << tree_node->state_ver.value() << "\n";
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DecisionTreeNodeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DecisionTreeNodeNode*>(ref.get());
      p->stream << "DecisionTree {" "\n";
      printDecisionTreeRecursively(p->stream, node);
      p->stream << "}" "\n";
    });

TVM_REGISTER_GLOBAL("auto_scheduler.DecisionTreeNode")
    .set_body_typed([](const Optional<PrimExpr>& predicate,
                       const Optional<DecisionTreeNode>& if_node,
                       const Optional<DecisionTreeNode>& else_node,
                       const Optional<StateVer>& state_ver) {
      if (predicate) {
        CHECK(if_node);
        CHECK(else_node);
        CHECK(!state_ver);
        return DecisionTreeNode(predicate.value(), if_node.value().operator->(),
                                else_node.value().operator->());
      }
      if (state_ver) {
        CHECK(!predicate);
        CHECK(!if_node);
        CHECK(!else_node);
        return DecisionTreeNode(state_ver.value());
      }
      LOG(FATAL) << "Neither predicate nor state_ver is defined";
      return DecisionTreeNode();
    });

TVM_REGISTER_NODE_TYPE(DecisionTreeNodeNode);

TVM_REGISTER_GLOBAL("driver.GenerateAndCompressIRMods")
    .set_body_typed([](const DynWklDispatcher& dyn_wkl_dispatcher,
                       const String& prefix) -> Array<ObjectRef> {
      return dyn_wkl_dispatcher->GenerateAndCompressIRMods(prefix);
    });

namespace {

Array<PrimExpr> ReplaceShapeVars(const Array<PrimExpr>& wkl_func_args,
                                 const Array<DynShapeVar>& shape_vars,
                                 const Array<PrimExpr>& new_shape_vars) {
  DynShapeVarReplacer replacer(
      [&shape_vars, &new_shape_vars](const DynShapeVarNode* op) -> PrimExpr {
        for (size_t i = 0; i < shape_vars.size(); ++i) {
          if (shape_vars[i]->name_hint == op->name_hint) {
            return new_shape_vars[i];
          }
        }
        LOG(FATAL) << "DynShapeVar=" << GetRef<DynShapeVar>(op)
                   << " has not been found in shape_vars";
        return GetRef<DynShapeVar>(op);
      });
  Array<PrimExpr> new_wkl_func_args;
  for (const PrimExpr arg : wkl_func_args) {
    new_wkl_func_args.push_back(replacer(arg));
  }
  return new_wkl_func_args;
}


Array<IntImm> InstantiateDynArgs(const Array<PrimExpr>& wkl_func_args,
                                 const Array<DynShapeVar>& shape_vars,
                                 const Array<IntImm>& wkl_inst) {
  DynShapeVarReplacer replacer(
      [&shape_vars, &wkl_inst](const DynShapeVarNode* op) -> PrimExpr {
        for (size_t i = 0; i < shape_vars.size(); ++i) {
          if (shape_vars[i]->name_hint == op->name_hint) {
            return wkl_inst[i];
          }
        }
        LOG(FATAL) << "DynShapeVar=" << GetRef<DynShapeVar>(op)
                   << " has not been found in shape_vars";
        return GetRef<DynShapeVar>(op);
      });
  Array<IntImm> instantiated_wkl_func_args;
  arith::Analyzer analyzer;
  for (const PrimExpr arg : wkl_func_args) {
    if (const IntImmNode* const arg_val =
          analyzer.Simplify(replacer(arg)).as<IntImmNode>()) {
      instantiated_wkl_func_args.push_back(Integer(arg_val->value));
    } else {
      LOG(FATAL) << "Unable to instantiate arg=" << arg << " with "
                    "shape_vars=" << shape_vars << " and wkl_inst=" << wkl_inst;
    }
  }
  return instantiated_wkl_func_args;
}

}  // namespace anonymous

TVM_REGISTER_GLOBAL("auto_scheduler.ReplaceShapeVars")
    .set_body_typed(ReplaceShapeVars);

TVM_REGISTER_GLOBAL("auto_scheduler.InstantiateDynArgs")
    .set_body_typed(InstantiateDynArgs);


}  // namespace auto_scheduler
}  // namespace tvm
