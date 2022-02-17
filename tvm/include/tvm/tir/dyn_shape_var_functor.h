// <bojian/DietCode>
#pragma once

#include <tvm/tir/dyn_shape_var.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>


namespace tvm {
namespace tir {

/**
 * @brief Replace the dynamic axis node with certain value.
 */
class DynShapeVarReplacer : public StmtExprMutator {
 private:
  std::function<PrimExpr(const DynShapeVarNode*)> freplace_expr_;
 protected:
  PrimExpr VisitExpr_(const DynShapeVarNode* op) override {
    if (freplace_expr_ == nullptr) {
      return GetRef<DynShapeVar>(op);
    } else {
      return freplace_expr_(op);
    }
  }
 public:
  explicit DynShapeVarReplacer(
      std::function<PrimExpr(const DynShapeVarNode*)> freplace_expr)
      : freplace_expr_(freplace_expr) {}
};

/**
 * @brief Find all the dynamic axis nodes of an expression.
 */
class DynShapeVarFinder : public StmtExprVisitor {
 public:
  std::unordered_set<const DynShapeVarNode*> dyn_shape_vars;
 protected:
  void VisitExpr_(const DynShapeVarNode* op) override {
    dyn_shape_vars.insert(op);
  }
};


std::pair<int, int> EvaluateRangeForAllWklInsts(
    const PrimExpr& expr, const Array<DynShapeVar>& shape_vars,
    const Array<Array<IntImm>>& wkl_insts);


/**
 * @brief Verify that a predicate holds true for all its dynamic axis nodes.
 */
bool canProveForAllDynShapeVars(arith::Analyzer& analyzer, PrimExpr predicate);

bool canProveForAllWklInsts(const PrimExpr& predicate,
                            const std::vector<PrimExpr>& constraint_stack,
                            const Map<Var, PrimExpr>& var_expr_map,
                            const Map<Var, Range>& var_range_map,
                            const Array<DynShapeVar>& shape_vars,
                            const Array<Array<IntImm>>& wkl_insts);

bool canProveForAllWklInsts(const PrimExpr& predicate,
                            const Map<IterVar, Range>& dom_map,
                            const Array<DynShapeVar>& shape_vars,
                            const Array<Array<IntImm>>& wkl_insts);

bool canProveLTForAllWklInsts(const PrimExpr& value, const PrimExpr& dom_extent,
                              const Map<IterVar, Range>& dom_map,
                              const Array<DynShapeVar>& shape_vars,
                              const Array<Array<IntImm>>& wkl_insts);

bool canProveGEForAllWklInsts(const PrimExpr& value, const PrimExpr& dom_extent,
                              const Map<IterVar, Range>& dom_map,
                              const Array<DynShapeVar>& shape_vars,
                              const Array<Array<IntImm>>& wkl_insts);


}  // namespace tir
}  // namespace tvm
