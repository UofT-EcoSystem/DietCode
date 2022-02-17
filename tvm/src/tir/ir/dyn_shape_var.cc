// <bojian/DietCodes>
#include <dmlc/parameter.h>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/dyn_shape_var.h>
#include <tvm/tir/dyn_shape_var_functor.h>
#include <tvm/tir/op.h>


namespace tvm {

namespace te {
extern bool enable_verbose_logging_in_msg_passing;
}


namespace tir {


DynShapeVar::DynShapeVar(String name
                         // , Array<IntImm> possible_values
                         ) {
  auto n = make_object<DynShapeVarNode>();
  n->name_hint = std::move(name);
  n->dtype = DataType::Int(32);
  // n->possible_values = std::move(possible_values);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.DynShapeVar")
    .set_body_typed([](String name
                       // , Array<IntImm> possible_values
                       ) {
      DynShapeVar ret(name
                      // , possible_values
                      );
      LOG(INFO) << ret;
      return ret;
    });


TVM_REGISTER_NODE_TYPE(DynShapeVarNode);


TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DynShapeVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const DynShapeVarNode*>(node.get());
      p->stream << "DynShapeVar(" << op->name_hint;
      // p->stream << " : [";
      // for (const IntImm& I : op->possible_values) {
      //   p->stream << I->value << ", ";
      // }
      // p->stream << "]";
      p->stream << ")";
    });


namespace {

/*
void canProveForEachDynShapeVar(arith::Analyzer& analyzer, PrimExpr predicate, bool* const can_prove,
                                const std::unordered_set<const DynShapeVarNode*>::iterator& dyn_shape_vars_iter,
                                const std::unordered_set<const DynShapeVarNode*>::iterator& dyn_shape_vars_end) {
  if (dyn_shape_vars_iter == dyn_shape_vars_end) {
    bool analyzer_result = analyzer.CanProve(predicate);
    if (!analyzer_result && dmlc::GetEnv("DIETCODE_DEBUG_TRACE", 0)) {
      LOG(WARNING) << "Unable to show that (" << predicate << ") is always true";
    }
    (*can_prove) &= analyzer_result;
    return;
  }
  const DynShapeVarNode* const dyn_shape_var = *dyn_shape_vars_iter;
  std::unordered_set<const DynShapeVarNode*>::iterator dyn_shape_vars_next_iter = dyn_shape_vars_iter;
  ++dyn_shape_vars_next_iter;

  for (const IntImm& v : dyn_shape_var->possible_values) {
    DynShapeVarReplacer dyn_shape_var_replacer(
        [dyn_shape_var, &v](const DynShapeVarNode* op) -> PrimExpr {
          if (op == dyn_shape_var) {
            return v;
          } else {
            return GetRef<DynShapeVar>(op);
          }
        });
    canProveForEachDynShapeVar(analyzer, dyn_shape_var_replacer(predicate), can_prove,
                               dyn_shape_vars_next_iter, dyn_shape_vars_end);
  }
}


Array<ObjectRef> InitializeDynShapeVars(
    const Array<PrimExpr>& args, const Array<DynShapeVar>& shape_vars,
    const Array<Array<IntImm>>& wkl_insts) {
  std::unordered_map<std::string, std::set<int>> dyn_shape_vars_info;
  for (const auto& wkl_inst : wkl_insts) {
    CHECK(shape_vars.size() == wkl_inst.size());
    for (size_t i = 0; i < shape_vars.size(); ++i) {
      dyn_shape_vars_info[std::string(shape_vars[i]->name_hint)].insert(wkl_inst[i]->value);
    }
  }
  Array<PrimExpr> new_args;
  std::unordered_map<std::string, DynShapeVar> dyn_shape_var_name_map;

  DynShapeVarReplacer dyn_shape_var_replacer(
      [&dyn_shape_vars_info, &dyn_shape_var_name_map](
          const DynShapeVarNode* op) -> PrimExpr {
        auto dyn_shape_vars_info_iter = dyn_shape_vars_info.find(std::string(op->name_hint));
        if (dyn_shape_vars_info_iter != dyn_shape_vars_info.end()) {

          auto dyn_shape_var_name_iter = dyn_shape_var_name_map.find(std::string(op->name_hint));
          if (dyn_shape_var_name_iter != dyn_shape_var_name_map.end()) {
            return dyn_shape_var_name_iter->second;
          }

          Array<IntImm> possible_values;
          
          for (const int v : dyn_shape_vars_info_iter->second) {
            possible_values.push_back(IntImm(DataType::Int(32), v));
          }
          DynShapeVar dyn_shape_var(dyn_shape_vars_info_iter->first, possible_values);
          dyn_shape_var_name_map[op->name_hint] = dyn_shape_var;
          return dyn_shape_var;
        } else {
          LOG(FATAL) << "DynShapeVar=" << GetRef<DynShapeVar>(op)
                     << " has not been found in shape_vars";
          return GetRef<DynShapeVar>(op);
        }
      });

  for (const PrimExpr& arg : args) {
    new_args.push_back(dyn_shape_var_replacer(arg));
  }
  Array<DynShapeVar> dyn_shape_vars;
  for (const DynShapeVar& var : shape_vars) {
    dyn_shape_vars.push_back(dyn_shape_var_name_map[var->name_hint]);
  }
  return {new_args, dyn_shape_vars};
}
 */

}  // namespace anonymous


std::pair<int, int> EvaluateRangeForAllWklInsts(
    const PrimExpr& expr, const Array<DynShapeVar>& shape_vars,
    const Array<Array<IntImm>>& wkl_insts) {
  if (const IntImmNode* const val = expr.as<IntImmNode>()) {
    return std::make_pair(val->value, val->value);
  }
  int min = std::numeric_limits<int>().max(),
      max = std::numeric_limits<int>().min();
  arith::Analyzer analyzer;
  for (const Array<IntImm>& wkl_inst : wkl_insts) {
    DynShapeVarReplacer dyn_shape_var_replacer(
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
    const IntImmNode* const val =
        analyzer.Simplify(dyn_shape_var_replacer(expr)).as<IntImmNode>();
    CHECK(val != nullptr);
    min = val->value < min ? val->value : min;
    max = val->value > max ? val->value : max;
  }
  return std::make_pair(min, max);
}


bool canProveForAllDynShapeVars(arith::Analyzer& analyzer, PrimExpr predicate) {
  DynShapeVarFinder dyn_shape_var_finder;
  // find all the dynamic axes within predicate
  dyn_shape_var_finder(predicate);
  bool can_prove = true;

  // canProveForEachDynShapeVar(analyzer, predicate, &can_prove,
  //                            dyn_shape_var_finder.dyn_shape_vars.begin(),
  //                            dyn_shape_var_finder.dyn_shape_vars.end());
  // return can_prove;
  // FIXME(bojian) This is a temporary workaround.
  // return !dyn_shape_var_finder.dyn_shape_vars.empty();
  return can_prove;
}

namespace {

bool EnterConstraintContext(const std::vector<PrimExpr>::iterator& it,
                            const std::vector<PrimExpr>::iterator& end,
                            arith::Analyzer* const analyzer,
                            const PrimExpr& predicate) {
  if (it == end) {
    return analyzer->CanProve(predicate);
  }
  With<arith::ConstraintContext> ctx(analyzer, *it);
  return EnterConstraintContext(it + 1, end, analyzer, predicate);
}

}

bool canProveForAllWklInsts(const PrimExpr& predicate,
                            const std::vector<PrimExpr>& constraint_stack,
                            const Map<Var, PrimExpr>& var_expr_map,
                            const Map<Var, Range>& var_range_map,
                            const Array<DynShapeVar>& shape_vars,
                            const Array<Array<IntImm>>& wkl_insts) {
  bool can_prove = true;
  for (const Array<IntImm>& wkl_inst : wkl_insts) {
    DynShapeVarReplacer dyn_shape_var_replacer(
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
    arith::Analyzer analyzer;

    for (const std::pair<Var, PrimExpr>& iv_expr_pair : var_expr_map) {
      analyzer.Bind(iv_expr_pair.first, dyn_shape_var_replacer(iv_expr_pair.second));
    }
    for (const std::pair<Var, Range>& iv_range_pair : var_range_map) {
      analyzer.Bind(
          iv_range_pair.first,
          Range::FromMinExtent(
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->min)),
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->extent))
          )
          );
      // LOG(INFO) << iv_range_pair.first << " : "
      //           << Range::FromMinExtent(
      //                analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->min)),
      //                analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->extent))
      //              );
    }

    PrimExpr instantiated_predicate = analyzer.Simplify(dyn_shape_var_replacer(predicate));

    std::vector<PrimExpr> instantiated_constraint_stack;
    for (const PrimExpr& constraint : constraint_stack) {
      instantiated_constraint_stack.push_back(analyzer.Simplify(dyn_shape_var_replacer(constraint)));
    }
    // LOG(INFO) << "predicate=" << predicate << ", "
    //              "instantiated_predicate=" << instantiated_predicate;

    can_prove &= // analyzer.CanProve(instantiated_predicate);
                 EnterConstraintContext(instantiated_constraint_stack.begin(),
                                        instantiated_constraint_stack.end(),
                                        &analyzer, instantiated_predicate);

    // if (!analyzer.CanProve(instantiated_predicate)) {
    //   LOG(WARNING) << "Cannot prove predicate=" << predicate << " -> "
    //                   "instantiated_predicate=" << instantiated_predicate; 
    //   LOG(INFO) << "predicate.type=" << instantiated_predicate->GetTypeKey();
    //   if (const LTNode* const lt_op = instantiated_predicate.as<LTNode>()) {
    //     LOG(INFO) << analyzer.int_set(lt_op->a, {}).max() << " vs. "<< lt_op->b;
    //   }
    // }
  }  // for (wkl_inst ∈ wkl_insts)
  return can_prove;
}


bool canProveForAllWklInsts(const PrimExpr& predicate,
                            const Map<IterVar, Range>& dom_map,
                            const Array<DynShapeVar>& shape_vars,
                            const Array<Array<IntImm>>& wkl_insts) {
  bool can_prove = true;
  for (const Array<IntImm>& wkl_inst : wkl_insts) {
    DynShapeVarReplacer dyn_shape_var_replacer(
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
    arith::Analyzer analyzer;

    for (const std::pair<IterVar, Range>& iv_range_pair : dom_map) {
      analyzer.Bind(
          iv_range_pair.first->var,
          Range::FromMinExtent(
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->min)),
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->extent))
          )
          );
    }
    PrimExpr instantiated_predicate = analyzer.Simplify(dyn_shape_var_replacer(predicate));

    // LOG(INFO) << "predicate=" << predicate << ", "
    //              "instantiated_predicate=" << instantiated_predicate;

    can_prove &= analyzer.CanProve(instantiated_predicate);
  }  // for (wkl_inst ∈ wkl_insts)
  return can_prove;
}


bool canProveLTForAllWklInsts(const PrimExpr& value, const PrimExpr& upper_bound,
                              const Map<IterVar, Range>& dom_map,
                              const Array<DynShapeVar>& shape_vars,
                              const Array<Array<IntImm>>& wkl_insts) {
  bool can_prove = true;
  for (const Array<IntImm>& wkl_inst : wkl_insts) {
    DynShapeVarReplacer dyn_shape_var_replacer(
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
    arith::Analyzer analyzer;
    Map<Var, arith::IntSet> iset_map;

    for (const std::pair<IterVar, Range>& iv_range_pair : dom_map) {
      Range iv_range =
          Range::FromMinExtent(
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->min)),
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->extent))
          );
      analyzer.Bind(iv_range_pair.first->var, iv_range);
      iset_map.Set(iv_range_pair.first->var,
                   arith::IntSet::FromRange(iv_range));
    }
    PrimExpr vmax = analyzer.int_set(dyn_shape_var_replacer(value), iset_map).max();
    can_prove &= analyzer.CanProve(vmax < dyn_shape_var_replacer(upper_bound));

    if (!analyzer.CanProve(vmax < dyn_shape_var_replacer(upper_bound)) &&
        te::enable_verbose_logging_in_msg_passing) {
      LOG(INFO) << "Cannot prove " << vmax << " < " << dyn_shape_var_replacer(upper_bound) << "!";
    }

  }  // for (wkl_inst ∈ wkl_insts)
  return can_prove;
}

bool canProveGEForAllWklInsts(const PrimExpr& value, const PrimExpr& lower_bound,
                              const Map<IterVar, Range>& dom_map,
                              const Array<DynShapeVar>& shape_vars,
                              const Array<Array<IntImm>>& wkl_insts) {
  bool can_prove = true;
  for (const Array<IntImm>& wkl_inst : wkl_insts) {
    DynShapeVarReplacer dyn_shape_var_replacer(
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
    arith::Analyzer analyzer;
    Map<Var, arith::IntSet> iset_map;

    for (const std::pair<IterVar, Range>& iv_range_pair : dom_map) {
      Range iv_range =
          Range::FromMinExtent(
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->min)),
            analyzer.Simplify(dyn_shape_var_replacer(iv_range_pair.second->extent))
          );
      analyzer.Bind(iv_range_pair.first->var, iv_range);
      iset_map.Set(iv_range_pair.first->var,
                   arith::IntSet::FromRange(iv_range));
    }
    PrimExpr vmin = analyzer.int_set(dyn_shape_var_replacer(value), iset_map).min();
    can_prove &= analyzer.CanProve(vmin >= dyn_shape_var_replacer(lower_bound));
  }  // for (wkl_inst ∈ wkl_insts)
  return can_prove;
}

/*
TVM_REGISTER_GLOBAL("auto_scheduler.InitializeDynShapeVars")
    .set_body_typed(
      [](const Array<PrimExpr>& args, const Array<DynShapeVar>& shape_vars,
         const Array<Array<IntImm>>& wkl_insts) {
        return InitializeDynShapeVars(args, shape_vars, wkl_insts);
      }
    );
 */

}  // namespace tir
}  // namespace tvm
