// <bojian/DietCode>
#pragma once

#include <tvm/tir/var.h>


namespace tvm {
namespace tir {


class DynShapeVarNode : public VarNode {
 public:
  // Array<IntImm> possible_values;

  void VisitAttrs(AttrVisitor* v) {
    VarNode::VisitAttrs(v);
    // v->Visit("possible_values", &possible_values);
  }

  static constexpr const char* _type_key = "tir.DynShapeVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(DynShapeVarNode, VarNode);
};


class DynShapeVar : public Var {
 public:
  TVM_DLL DynShapeVar(String name
                      // , Array<IntImm> possible_values
                      );

  TVM_DEFINE_OBJECT_REF_METHODS(DynShapeVar, Var, DynShapeVarNode);
};


}  // namespace tir
}  // namespace tvm
