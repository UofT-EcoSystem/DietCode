#pragma once

#include <tvm/auto_scheduler/search_task.h>
#include <tvm/runtime/object.h>
#include <tvm/ir/expr.h>


namespace tvm {
namespace auto_scheduler {


class StateVerNode : public Object {
 public:
  Integer major, minor;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("major", &major);
    v->Visit("minor", &minor);
  }

  bool SEqualReduce(const StateVerNode* other, SEqualReducer equal) const {
    return equal(major, other->major) && equal(minor, other->minor);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(major);
    hash_reduce(minor);
  }

  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const char* _type_key = "auto_scheduler.StateVer";
  TVM_DECLARE_FINAL_OBJECT_INFO(StateVerNode, Object);
};

class StateVer : public ObjectRef {
 public:
  StateVer(const int major, const int minor);
  TVM_DEFINE_OBJECT_REF_METHODS(StateVer, ObjectRef, StateVerNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateVerNode);
};


class DynWklDispatcherNode : public Object {
 public:
  SearchTask search_task;
  std::vector<State> states;
  std::unordered_map<size_t, size_t> inst_disp_map;
  // Map<Array<IntImm>, Integer> wkl_inst_func_gv_map;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("search_task", &search_task);
    // v->Visit("states", &states);
    // v->Visit("inst_disp_map", &inst_disp_map);
  }
  Array<ObjectRef> Dispatch(const int wkl_idx) const;
  State DispatchToState(const int wkl_idx) const;
  // Array<ObjectRef> GetSkeleton() const;
  IRModule GetSkeleton(const String& name) const;
  Array<ObjectRef> GenerateAndCompressIRMods(const String& prefix) const;

  void EmbedComputeDAG(const ComputeDAG& compute_dag);
  static constexpr const char* _type_key = "auto_scheduler.DynWklDispatcher";
  TVM_DECLARE_FINAL_OBJECT_INFO(DynWklDispatcherNode, Object);
};

class DynWklDispatcher : public ObjectRef {
 public:
  DynWklDispatcher(const SearchTask& search_task, std::vector<State>&& states,
                   std::unordered_map<size_t, size_t>&& inst_disp_map);
  TVM_DEFINE_OBJECT_REF_METHODS(DynWklDispatcher, ObjectRef,
                                DynWklDispatcherNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DynWklDispatcherNode);
};


class DecisionTreeNodeNode : public Object {
 public:
  Optional<PrimExpr> predicate = Optional<PrimExpr>(nullptr);
  const DecisionTreeNodeNode* if_node = nullptr, * else_node = nullptr;
  Optional<StateVer> state_ver = Optional<StateVer>(nullptr);

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("predicate", &predicate);
    v->Visit("state_ver", &state_ver);
  }

  static constexpr const char* _type_key = "auto_scheduler.DecisionTreeNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecisionTreeNodeNode, Object);
};


class DecisionTreeNode : public ObjectRef {
 public:
  DecisionTreeNode(const StateVer& state_ver);
  DecisionTreeNode(const PrimExpr& predicate,
                   const DecisionTreeNodeNode* const if_node,
                   const DecisionTreeNodeNode* const else_node);

  TVM_DEFINE_OBJECT_REF_METHODS(DecisionTreeNode, ObjectRef,
                                DecisionTreeNodeNode);
};


}  // namespace auto_scheduler
}  // namespace tvm
