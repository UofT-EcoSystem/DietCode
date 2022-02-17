/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/arith/ir_mutator_with_analyzer.cc
 */
#include "ir_mutator_with_analyzer.h"

#include <tvm/tir/analysis.h>

// <bojian/DietCode>
#include <tvm/tir/dyn_shape_var_functor.h>

#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

using namespace tir;

Stmt IRMutatorWithAnalyzer::VisitStmt_(const ForNode* op) {
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));

  // <bojian/DietCode>
  var_range_bind_map_.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));

  return StmtExprMutator::VisitStmt_(op);
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const LetStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    analyzer_->Bind(op->var, value);

    // <bojian/DietCode>
    var_expr_bind_map_.Set(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr real_condition = condition;
  static auto op_likely = Op::Get("tir.likely");

  if (auto call = condition.as<CallNode>()) {
    if (call->op.same_as(op_likely)) {
      real_condition = call->args[0];
    }
  }

  Stmt then_case, else_case;
  {
    With<ConstraintContext> ctx(analyzer_, real_condition);

    // <bojian/DietCode>
    constraint_stack_.push_back(real_condition);

    then_case = this->VisitStmt(op->then_case);

    // <bojian/DietCode>
    constraint_stack_.pop_back();
  }
  if (op->else_case.defined()) {
    With<ConstraintContext> ctx(analyzer_, analyzer_->rewrite_simplify(Not(real_condition)));

    // <bojian/DietCode>
    constraint_stack_.push_back(analyzer_->rewrite_simplify(Not(real_condition)));

    else_case = this->VisitStmt(op->else_case);

    // <bojian/DietCode>
    constraint_stack_.pop_back();
  }

  if (is_one(real_condition)) {
    // LOG(INFO) << "op->condition=" << op->condition << ", "
    //              "condition=" << condition << ", "
    //              "real_condition=" << real_condition << " "
    //              "returning the then case";
    // LOG(INFO) << "CanProve=" << std::boolalpha << analyzer_->CanProve(op->condition)
    //           << std::noboolalpha;
    // LOG(INFO) << "var_expr_bind_map=" << var_expr_bind_map_ << ", "
    //              "var_range_bind_map=" << var_range_bind_map_;
    // if (auto call_op = op->condition.as<CallNode>()) {
    //   if (call_op->op.same_as(op_likely)) {
    //     if (const LTNode* const lt_op = call_op->args[0].as<LTNode>()) {
    //       LOG(INFO) << analyzer_->int_set(lt_op->a, {}).max() << " vs. "<< lt_op->b;
    //       LOG(INFO) << analyzer_->const_int_bound(analyzer_->rewrite_simplify(lt_op->a));
    //     }
    //   }
    // }

    return then_case;
  }

    // <bojian/DietCode>
    else if (!shape_vars.empty() && !wkl_insts.empty() &&
             canProveForAllWklInsts(real_condition, constraint_stack_,
                                    var_expr_bind_map_, var_range_bind_map_,
                                    shape_vars, wkl_insts)) {
    // LOG(INFO) << "shape_vars=" << shape_vars << ", wkl_insts=" << wkl_insts;
    // LOG(INFO) << "op->condition=" << op->condition << ", "
    //              "condition=" << condition << ", "
    //              "real_condition=" << real_condition << " "
    //              "returning the then case";

    return then_case;
  }

  if (is_zero(real_condition)) {
    if (else_case.defined()) {
      return else_case;
    }
    return Evaluate(0);
  }

  if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread) {
    IterVar iv = Downcast<IterVar>(op->node);
    ICHECK_NE(iv->thread_tag.length(), 0U);
    analyzer_->Bind(iv->var, Range::FromMinExtent(0, op->value));

    // <bojian/DietCode>
    var_range_bind_map_.Set(iv->var, Range::FromMinExtent(0, op->value));

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    return stmt;
  } else {
    return StmtExprMutator::VisitStmt_(op);
  }
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const AssertStmtNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr message = this->VisitExpr(op->message);
  With<ConstraintContext> ctx(analyzer_, condition);

  // <bojian/DietCode>
  constraint_stack_.push_back(condition);

  Stmt body = this->VisitStmt(op->body);

  // <bojian/DietCode>
  constraint_stack_.pop_back();

  if (condition.same_as(op->condition) && message.same_as(op->message) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->message = std::move(message);
    n->body = std::move(body);
    return Stmt(n);
  }
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const CallNode* op) {
  // add condition context to if_then_else
  static auto op_if_then_else = Op::Get("tir.if_then_else");
  if (op->op.same_as(op_if_then_else)) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    PrimExpr true_value, false_value;
    {
      With<ConstraintContext> constraint(analyzer_, cond);

      // <bojian/DietCode>
      constraint_stack_.push_back(cond);

      true_value = this->VisitExpr(op->args[1]);

      // <bojian/DietCode>
      constraint_stack_.pop_back();
    }
    {
      With<ConstraintContext> constraint(analyzer_, analyzer_->rewrite_simplify(Not(cond)));

      // <bojian/DietCode>
      constraint_stack_.push_back(analyzer_->rewrite_simplify(Not(cond)));

      false_value = this->VisitExpr(op->args[2]);

      // <bojian/DietCode>
      constraint_stack_.pop_back();
    }
    if (is_zero(cond)) {
      return false_value;
    }
    if (is_one(cond)) {
      return true_value;
    }
    if (cond.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      return Call(op->dtype, op->op, {cond, true_value, false_value});
    }
  }
  return StmtExprMutator::VisitExpr_(op);
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const LetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    analyzer_->Bind(op->var, value);

    // <bojian/DietCode>
    var_expr_bind_map_.Set(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Let(op->var, value, body);
  }
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const SelectNode* op) {
  PrimExpr cond = this->VisitExpr(op->condition);
  PrimExpr true_value, false_value;
  {
    With<ConstraintContext> constraint(analyzer_, cond);

    // <bojian/DietCode>
    constraint_stack_.push_back(cond);

    true_value = VisitExpr(op->true_value);

    // <bojian/DietCode>
    constraint_stack_.pop_back();
  }
  {
    With<ConstraintContext> constraint(analyzer_, analyzer_->rewrite_simplify(Not(cond)));

    // <bojian/DietCode>
    constraint_stack_.push_back(analyzer_->rewrite_simplify(Not(cond)));

    false_value = VisitExpr(op->false_value);

    // <bojian/DietCode>
    constraint_stack_.pop_back();
  }
  if (is_zero(cond)) {
    return false_value;
  }
  if (is_one(cond)) {
    return true_value;
  }
  // normal path
  if (cond.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Select(cond, true_value, false_value);
  }
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const ReduceNode* op) {
  // Setup the domain information before simplification.
  for (const IterVar& iv : op->axis) {
    analyzer_->Bind(iv->var, iv->dom);

    // <bojian/DietCode>
    var_range_bind_map_.Set(iv->var, iv->dom);
  }
  // Recursively call simplification when necessary.
  return StmtExprMutator::VisitExpr_(op);
}

}  // namespace arith
}  // namespace tvm
