#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/auto_scheduler/dietcode.h>
#include "./utils.h"


namespace tvm {
namespace auto_scheduler {
namespace {

/*
class InlineDispatchAnalysis : public StmtExprVisitor {
 private:
  DynWklDispatcher dyn_wkl_dispatcher_;
 public:
  int num_postproc_args, num_tensor_args = -1;

  explicit InlineDispatchAnalysis(const DynWklDispatcher& dyn_wkl_dispatcher)
      : dyn_wkl_dispatcher_(dyn_wkl_dispatcher) {}

  void VisitExpr_(const CallNode* op) override final {
    if (!op->op.defined()) {
      return;
    }
    if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
      String func_name = Downcast<StringImm>(op->args[0])->value;
      if (func_name == "__tvm_set_device") {
        return;
      }
      int num_args = Downcast<IntImm>(op->args[op->args.size() - 1])->value;
      num_postproc_args =
          num_args - dyn_wkl_dispatcher_->search_task->shape_vars.value().size();
    }
    if (op->op.same_as(builtin::tvm_struct_set())) {
      PrimExpr last_arg = op->args[3];
      if (const CastNode* const cast_op = last_arg.as<CastNode>()) {
        if (const DynShapeVarNode* const dyn_shape_var_op =
              cast_op->value.as<DynShapeVarNode>()) {
          if (num_tensor_args == -1) {
            num_tensor_args = Downcast<IntImm>(op->args[1])->value;
          }
        }
      }
    }
  }
};


static inline int ExtractFirstIntFromString(const std::string& str) {
  static const char* const digits = "0123456789";
  const std::size_t first_digit_pos = str.find_first_of(digits);
  if (first_digit_pos != std::string::npos) {
    const std::size_t first_non_digit_pos_after_digit = str.find_first_not_of(digits, first_digit_pos);
    return std::stoi(
             str.substr(first_digit_pos,
                        first_non_digit_pos_after_digit != std::string::npos ?
                          first_non_digit_pos_after_digit - first_digit_pos :
                          first_non_digit_pos_after_digit)
                    );
  }
  return -1;
}


static inline int ExtractLastIntFromString(const std::string& str) {
  static const char* const digits = "0123456789";
  size_t last_index = str.find_last_not_of(digits);
  return std::stoi(str.substr(last_index + 1));
}


class InlineDispatchTransform : public StmtExprMutator {
 private:
  InlineDispatchAnalysis analysis_;
  IRModule merged_mod_dev_;
  DynWklDispatcher dyn_wkl_dispatcher_;
  Array<Var> shape_vars_;

  PrimExpr GetPredicate(const Array<IntImm>& wkl_inst) {
    CHECK(shape_vars_.size() > 0);
    PrimExpr ret = operator==(shape_vars_[0], wkl_inst[0]);
    for (size_t i = 1; i < shape_vars_.size(); ++i) {
      ret = (ret && operator==(shape_vars_[i], wkl_inst[i]));
    }
    return ret;
  }

  void Dispatch(Array<Stmt>* const new_body, const Array<Stmt>& orig_body,
                const Call& kernel_func_call, const int new_arg_start,
                const int orig_arg_start) {
    BlockIdxExtractor blockIdx_extractor;
    ThreadIdxExtractor threadIdx_extractor;
    const Map<GlobalVar, BaseFunc>& kernel_funcs = merged_mod_dev_->functions;

    for (size_t wkl_inst_id = 0;
         wkl_inst_id < dyn_wkl_dispatcher_->search_task->wkl_insts.size();
         ++wkl_inst_id) {
      const Array<IntImm>& wkl_inst =
          dyn_wkl_dispatcher_->search_task->wkl_insts[wkl_inst_id];
      auto kernel_func_it =
          std::find_if(kernel_funcs.begin(), kernel_funcs.end(),
                       [&wkl_inst_id](
                           const std::pair<GlobalVar, BaseFunc>& gv_func_pair) {
                         std::string func_name =
                             std::string(gv_func_pair.first->name_hint.c_str());
                         std::size_t kernel_pos = func_name.find_last_of("_kernel");
                         func_name = func_name.substr(0, kernel_pos - 6);
                         int func_name_id = ExtractLastIntFromString(func_name);
                         return func_name_id == static_cast<int>(wkl_inst_id);
                       });
      CHECK(kernel_func_it != kernel_funcs.end())
          << "wkl_inst_id=" << wkl_inst_id << " has not been found in "
          << MapToString(kernel_funcs);

      PrimFunc func = Downcast<PrimFunc>((*kernel_func_it).second);
      blockIdx_extractor(func->body);
      threadIdx_extractor(func->body);
      PrimExpr blockIdx_x = blockIdx_extractor.blockIdx_x_ext,
               threadIdx_x = threadIdx_extractor.threadIdx_x_ext;
      
      Call stack_set_value =
          Downcast<Call>(Downcast<Evaluate>(orig_body[2 * orig_arg_start])->value);
      Store stack_set_tcode =
          Downcast<Store>(orig_body[2 * orig_arg_start + 1]);
      Array<Stmt> disp_body;
      for (int new_arg_idx = new_arg_start, orig_arg_idx = orig_arg_start;
           new_arg_idx < analysis_.num_postproc_args;
           ++new_arg_idx, ++orig_arg_idx) {
        PrimExpr value_to_set =
            new_arg_idx == new_arg_start ? blockIdx_x : threadIdx_x;
        disp_body.push_back(Evaluate(Call(stack_set_value->dtype,
                                          tir::builtin::tvm_struct_set(),
                                          {stack_set_value->args[0],
                                           Integer(new_arg_idx),
                                           stack_set_value->args[2],
                                           Cast(DataType::Int(64), value_to_set)
                                           }
                                          )
                                     )
                            );
        disp_body.push_back(Store(stack_set_tcode->buffer_var,
                                  stack_set_tcode->value,
                                  Integer(new_arg_idx),
                                  stack_set_tcode->predicate
                                  )
                            );
      }
      disp_body.push_back(Evaluate(Call(kernel_func_call->dtype,
                                        tir::builtin::tvm_call_packed_lowered(),
                                        {StringImm((*kernel_func_it).first->name_hint),
                                         kernel_func_call->args[1],  // stack_value
                                         kernel_func_call->args[2],  // stack_tcode,
                                         kernel_func_call->args[3],  // 0,
                                         analysis_.num_postproc_args
                                        }
                                        )
                                   )
                          );
      new_body->push_back(IfThenElse(GetPredicate(wkl_inst),
                                     SeqStmt(disp_body)));
    }
  }

 public:
  InlineDispatchTransform(const InlineDispatchAnalysis& analysis,
                          const IRModule& merged_mod_dev,
                          const DynWklDispatcher& dyn_wkl_dispatcher)
      : analysis_(analysis), merged_mod_dev_(merged_mod_dev),
        dyn_wkl_dispatcher_(dyn_wkl_dispatcher) {}
  Stmt VisitStmt_(const LetStmtNode* op) override final {
    // resize the stack allocations
    if (op->var->name_hint == "stack_tcode" || 
        op->var->name_hint == "stack_value") {
      Call stack_initializer = Downcast<Call>(op->value);
      return LetStmt(op->var, Call(stack_initializer->dtype,
                                   tir::builtin::tvm_stack_alloca(),
                                   {stack_initializer->args[0],
                                    analysis_.num_postproc_args}
                                   ),
                     VisitStmt(op->body));
    }
    const Array<DynShapeVar>& search_task_shape_vars =
        dyn_wkl_dispatcher_->search_task->shape_vars.value();
    auto shape_var_it =
        std::find_if(search_task_shape_vars.begin(), search_task_shape_vars.end(),
                     [&op](const DynShapeVar& dyn_shape_var) {
                       return dyn_shape_var->name_hint.compare(op->var->name_hint) == 0;
                     });
    if (shape_var_it != search_task_shape_vars.end()) {
      shape_vars_.push_back(op->var);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const AttrStmtNode* op) override final {
    if (op->attr_key == "compute_scope") {
      SeqStmt orig_body = Downcast<SeqStmt>(op->body);
      Array<Stmt> new_body;
      int orig_arg_idx = 0;
      for (orig_arg_idx = 0; orig_arg_idx < analysis_.num_tensor_args;
           ++orig_arg_idx) {
        new_body.push_back(orig_body->seq[2 * orig_arg_idx]);      // stack_value
        new_body.push_back(orig_body->seq[2 * orig_arg_idx + 1]);  // stack_tcode
      }
      int new_arg_idx = orig_arg_idx;
      for (size_t i = 0;
           i < dyn_wkl_dispatcher_->search_task->shape_vars.value().size();
           ++i, ++orig_arg_idx) {}
      Call kernel_func_call =
          Downcast<Call>(Downcast<Evaluate>(
            orig_body->seq[orig_body->seq.size() - 1])->value);
      Dispatch(&new_body, orig_body->seq, kernel_func_call, new_arg_idx,
               orig_arg_idx);
      return AttrStmt(op->node, op->attr_key, op->value, SeqStmt(new_body));
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};
 */

class ShapeVarReplacer : public StmtExprMutator {
 private:
  Array<Var> shape_vars_;
 public:
  ShapeVarReplacer(const Array<Var>& shape_vars) : shape_vars_(shape_vars) {}
 protected:
  PrimExpr VisitExpr_(const VarNode* op) override {
    for (size_t i = 0; i < this->shape_vars_.size(); ++i) {
      if (this->shape_vars_[i]->name_hint == op->name_hint) {
        return this->shape_vars_[i];
      }
    }
    LOG(FATAL) << "Var=" << GetRef<Var>(op) << " has not been found in "
                  "shape_vars=" << shape_vars_;
    return GetRef<Var>(op);
  }
  PrimExpr VisitExpr_(const DynShapeVarNode* op) override {
    for (size_t i = 0; i < this->shape_vars_.size(); ++i) {
      if (this->shape_vars_[i]->name_hint == op->name_hint) {
        return this->shape_vars_[i];
      }
    }
    LOG(FATAL) << "DynShapeVar=" << GetRef<DynShapeVar>(op) << " has not been found in "
                  "shape_vars=" << shape_vars_;
    return GetRef<DynShapeVar>(op);
  }
};

class ArgsExtractor : public StmtExprVisitor {
 private:
  std::string func_name_;
 public:
  Optional<Array<PrimExpr>> call_args;
  ArgsExtractor(const std::string& func_name) : func_name_(func_name) {}
 protected:
  void VisitExpr_(const CallNode* op) override final {
    if (!op->args.defined() || op->args.size() == 0) {
      return;
    }
    if (const StringImmNode* const func_name = op->args[0].as<StringImmNode>()) {
      if (func_name_ == std::string(func_name->value)) {
        call_args = op->args;
      }
    }
  }
};

class ArgsReplacer : public ExprMutator {
 private:
  Array<PrimExpr> orig_args_;
 public:
  ArgsReplacer(const Array<PrimExpr>& orig_args) : orig_args_(orig_args) {}
 protected:
  PrimExpr VisitExpr_(const VarNode* op) override final {
    for (const PrimExpr& arg : orig_args_) {
      if (const VarNode* const vnode = arg.as<VarNode>()) {
        if (op->name_hint == vnode->name_hint){
           return arg;
        }
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const DynShapeVarNode* op) override final {
    for (const PrimExpr& arg : orig_args_) {
      if (const DynShapeVarNode* const vnode = arg.as<DynShapeVarNode>()) {
        if (op->name_hint == vnode->name_hint){
           return arg;
        }
      }
    }
    return ExprMutator::VisitExpr_(op);
  }
};


class InlineDispatchTransform : public StmtExprMutator {
 private:
  Array<te::Tensor> tensors_;
  Array<DynShapeVar> orig_shape_vars_;
  DecisionTreeNode tree_classifier_root_;
  IRModule merged_mod_host_;
  Array<Var> shape_vars_;
  String name_prefix_;

 private:

  Array<PrimExpr> LocateCallArgs(const std::string& func_name) const {
    ArgsExtractor extractor(func_name);
    for (const auto& gv_func_pair : merged_mod_host_->functions) {
      extractor(Downcast<PrimFunc>(gv_func_pair.second)->body);
      if (extractor.call_args) {
        return extractor.call_args.value();
      }
    }
    LOG(FATAL) << "func_name=" << func_name << " not found";
    return Array<PrimExpr>();
  }

  Stmt Dispatch(const DecisionTreeNodeNode* const tree_node,
                ShapeVarReplacer& shape_var_replacer,
                const CallNode* const orig_call_op,
                const std::string& name_prefix,
                const std::string& name_suffix) const {
    if (tree_node->predicate) {
      CHECK(tree_node->if_node);
      CHECK(tree_node->else_node);
      return IfThenElse(shape_var_replacer(tree_node->predicate.value()),
                        Dispatch(tree_node->if_node, shape_var_replacer, orig_call_op,
                                 name_prefix, name_suffix),
                        Dispatch(tree_node->else_node, shape_var_replacer, orig_call_op,
                                 name_prefix, name_suffix));
    } else {
      CHECK(tree_node->state_ver);
      std::string func_name = 
          name_prefix + "_" + std::to_string(tree_node->state_ver.value()->major->value)
                      + "_" + std::to_string(tree_node->state_ver.value()->minor->value) +
          name_suffix;
      Array<PrimExpr> merged_mod_host_args = LocateCallArgs(func_name),
                      new_args;
                      // new_args = orig_call_op->args;
      ArgsReplacer args_replacer(orig_call_op->args);

      // CHECK(new_args.size() == merged_mod_host_args.size())
      //     << "new_args=" << new_args << " does not match that of "
      //        "merged_mod_host_args=" << merged_mod_host_args;
      // new_args.Set(0, StringImm(func_name));
      // for (size_t i = tensors_.size() + orig_shape_vars_.size() + 1;
      //      i < new_args.size(); ++i) {
      //   new_args.Set(i, replacer(merged_mod_host_args[i]));
      // }
      new_args.push_back(StringImm(func_name));
      for (size_t i = 1; i < merged_mod_host_args.size(); ++i) {
        new_args.push_back(args_replacer(merged_mod_host_args[i]));
      }

      return Evaluate(Call(orig_call_op->dtype,
                           orig_call_op->op,
                           new_args));
    }
  }

 public:
  InlineDispatchTransform(const Array<te::Tensor>& tensors,
                          const Array<DynShapeVar>& shape_vars,
                          const DecisionTreeNode& tree_classifier_root,
                          const IRModule& merged_mod_host,
                          const String& name_prefix)
      : tensors_(tensors), orig_shape_vars_(shape_vars),
        tree_classifier_root_(tree_classifier_root),
        merged_mod_host_(merged_mod_host),
        name_prefix_(name_prefix)
        {}
  Stmt VisitStmt_(const LetStmtNode* op) override final {
    auto shape_var_it =
        std::find_if(orig_shape_vars_.begin(), orig_shape_vars_.end(),
                     [&op](const DynShapeVar& dyn_shape_var) {
                       return dyn_shape_var->name_hint.compare(op->var->name_hint) == 0;
                     });
    if (shape_var_it != orig_shape_vars_.end()) {
      shape_vars_.push_back(op->var);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const EvaluateNode* op) override final {
    if (const CallNode* const call_op = op->value.as<CallNode>()) {
      std::string func_name = Downcast<StringImm>(call_op->args[0])->value;
      size_t name_prefix_pos = func_name.find(name_prefix_);
      if (name_prefix_pos == std::string::npos) {
        return StmtExprMutator::VisitStmt_(op);
      }
      ShapeVarReplacer replacer(shape_vars_);
      return Dispatch(tree_classifier_root_.get(), replacer, call_op,
                      func_name.substr(0, name_prefix_pos + name_prefix_.size()),
                      func_name.substr(name_prefix_pos + name_prefix_.size())
                      );
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};


}  // anonymous namespace

IRModule InlineDispatch(// IRModule skeleton_mod_host,
                        // const IRModule& merged_mod_dev,
                        // const DynWklDispatcher& dyn_wkl_dispatcher
                        const Array<te::Tensor>& tensors,
                        const Array<DynShapeVar>& shape_vars,
                        const DecisionTreeNode& tree_classifier_root,
                        IRModule skeleton_mod_host,
                        const IRModule& merged_mod_host,
                        const IRModule& merged_mod_dev,
                        const String& name_prefix
                        ) {
  // InlineDispatchAnalysis analysis(dyn_wkl_dispatcher);

  IRModuleNode* const mutable_skeleton_mod_host =
      skeleton_mod_host.CopyOnWrite();
  MapNode* const mutable_skeleton_mod_host_funcs =
      mutable_skeleton_mod_host->functions.CopyOnWrite();

  for (auto& gv_basef_pair : *mutable_skeleton_mod_host_funcs) {
    if (const PrimFuncNode* const primf_node =
          gv_basef_pair.second.as<PrimFuncNode>()) {
      PrimFunc primf_ref = GetRef<PrimFunc>(primf_node);
      // analysis(primf_node->body);

      // LOG(INFO) << "num_postproc_args=" << analysis.num_postproc_args << ", "
      //              "num_tensor_args=" << analysis.num_tensor_args;
      // LOG(INFO) << "body=" << primf_node->body;
      // InlineDispatchTransform transform(analysis, merged_mod_dev,
      //                                   dyn_wkl_dispatcher
      //                                   );
      InlineDispatchTransform transform(tensors, shape_vars,
                                        tree_classifier_root,
                                        merged_mod_host,
                                        name_prefix);
      PrimFuncNode* const mutable_primf = primf_ref.CopyOnWrite();
      mutable_primf->body = transform(primf_node->body);

      // LOG(INFO) << "Transform completed";
      gv_basef_pair.second = std::move(primf_ref);
    }
  }
  return skeleton_mod_host;
}

TVM_REGISTER_GLOBAL("auto_scheduler.InlineDispatch").set_body_typed(InlineDispatch);

}  // namespace auto_scheduler


namespace tir {
namespace transform {

Pass MarkAlreadyOptFlag() {
  auto pass_func =
      [](PrimFunc f, IRModule m, PassContext ctx) {
        // Optional<Bool> is_entry_func_attr =
        //     f->GetAttr<Bool>("tir.is_entry_func");
        // if (!is_entry_func_attr.defined()) {
        //   return f;
        // }
        auto* func_node = f.CopyOnWrite();
        DictAttrsNode* const attr_dict = func_node->attrs.CopyOnWrite();
        attr_dict->dict.Set(String("already_opt"), Bool(true));
        return f;
      };
  return CreatePrimFuncPass(pass_func, 0, "tir.MarkAlreadyOptFlag", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MarkAlreadyOptFlag").set_body_typed(MarkAlreadyOptFlag);

}  // namespace transform
}  // namespace tir

}  // namespace tvm
