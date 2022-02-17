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
 * \file auto_scheduler/measure.cc
 * \brief Distributed measurement infrastructure to measure the runtime costs of tensor programs.
 */

#include <tvm/auto_scheduler/measure.h>
#include <tvm/runtime/registry.h>

// <bojian/DietCode>
#include <tvm/support/parallel_for.h>

#include <algorithm>

#include "search_policy/empty_policy.h"
#include "search_policy/sketch_policy.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(MeasureInputNode);
TVM_REGISTER_NODE_TYPE(BuildResultNode);
TVM_REGISTER_NODE_TYPE(MeasureResultNode);
TVM_REGISTER_OBJECT_TYPE(MeasureCallbackNode);
TVM_REGISTER_OBJECT_TYPE(PythonBasedMeasureCallbackNode);
TVM_REGISTER_OBJECT_TYPE(ProgramRunnerNode);
TVM_REGISTER_OBJECT_TYPE(ProgramBuilderNode);
TVM_REGISTER_OBJECT_TYPE(ProgramMeasurerNode);
TVM_REGISTER_OBJECT_TYPE(LocalBuilderNode);
TVM_REGISTER_OBJECT_TYPE(LocalRunnerNode);
TVM_REGISTER_OBJECT_TYPE(RPCRunnerNode);

static const char* ErrorNoToStr[] = {
    "NoError",
    "InstantiationError",
    "CompileHostError",
    "CompileDeviceError",
    "RuntimeDeviceError",
    "WrongAnswerError",
    "BuildTimeoutError",
    "RunTimeoutError",
    "UnknownError",
};

/********** Measure input and result **********/
MeasureInput::MeasureInput(SearchTask task, State state

                           // <bojian/DietCode>
                         , Optional<Array<IntImm>> wkl_inst

                           ) {
  auto node = make_object<MeasureInputNode>();
  node->task = std::move(task);
  node->state = std::move(state);

  // <bojian/DietCode>
  node->wkl_inst = std::move(wkl_inst);

  data_ = std::move(node);
}

MeasureInput MeasureInputNode::copy() const {
  auto node = make_object<MeasureInputNode>();
  node->task = task;
  node->state = state;
  return MeasureInput(node);
}

BuildResult::BuildResult(String filename, Array<te::Tensor> args, int error_no, String error_msg,
                         double time_cost) {
  auto node = make_object<BuildResultNode>();
  node->filename = std::move(filename);
  node->args = std::move(args);
  node->error_no = error_no;
  node->error_msg = std::move(error_msg);
  node->time_cost = time_cost;
  data_ = std::move(node);
}

MeasureResult::MeasureResult(Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
                             double timestamp) {
  auto node = make_object<MeasureResultNode>();
  node->costs = std::move(costs);
  node->error_no = error_no;
  node->error_msg = std::move(error_msg);
  node->all_cost = all_cost;
  node->timestamp = timestamp;
  data_ = std::move(node);
}

MeasureResult MeasureResultNode::copy() const {
  auto node = make_object<MeasureResultNode>();
  node->costs = costs;
  node->error_no = error_no;
  node->error_msg = error_msg;
  node->all_cost = all_cost;
  node->timestamp = timestamp;
  return MeasureResult(node);
}

/********** LocalBuilder **********/
LocalBuilder::LocalBuilder(int timeout, int n_parallel, const String& build_func) {
  auto node = make_object<LocalBuilderNode>();
  node->timeout = timeout;
  node->n_parallel = n_parallel;
  node->build_func = build_func;
  data_ = std::move(node);
}

Array<BuildResult> LocalBuilderNode::Build(const Array<MeasureInput>& inputs, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.local_builder.build")) {
    Array<BuildResult> results = (*f)(inputs, timeout, n_parallel, build_func, verbose);
    return results;
  }
  LOG(FATAL) << "auto_scheduler.local_builder.build is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** LocalRunner **********/
LocalRunner::LocalRunner(int timeout, int number, int repeat, int min_repeat_ms,
                         double cooldown_interval, bool enable_cpu_cache_flush) {
  ObjectPtr<LocalRunnerNode> node = make_object<LocalRunnerNode>();
  node->timeout = timeout;
  node->number = number;
  node->repeat = repeat;
  node->min_repeat_ms = min_repeat_ms;
  node->cooldown_interval = cooldown_interval;
  node->enable_cpu_cache_flush = enable_cpu_cache_flush;
  data_ = std::move(node);
}

Array<MeasureResult> LocalRunnerNode::Run(const Array<MeasureInput>& inputs,
                                          const Array<BuildResult>& build_results, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.local_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, timeout, number, repeat, min_repeat_ms, cooldown_interval,
             enable_cpu_cache_flush, verbose);
    return results;
  }
  LOG(FATAL) << "auto_scheduler.local_runner.run is not registered. "
             << "This is a function registered in Python, "
             << "make sure the TVM Python runtime has been loaded successfully.";
  throw;
}

/********** RPCRunner **********/
RPCRunner::RPCRunner(const String& key, const String& host, int port, int priority, int n_parallel,
                     int timeout, int number, int repeat, int min_repeat_ms,
                     double cooldown_interval, bool enable_cpu_cache_flush) {
  auto node = make_object<RPCRunnerNode>();
  node->key = key;
  node->host = host;
  node->port = port;
  node->priority = priority;
  node->timeout = timeout;
  node->n_parallel = n_parallel;
  node->number = number;
  node->repeat = repeat;
  node->min_repeat_ms = min_repeat_ms;
  node->cooldown_interval = cooldown_interval;
  node->enable_cpu_cache_flush = enable_cpu_cache_flush;
  data_ = std::move(node);
}

Array<MeasureResult> RPCRunnerNode::Run(const Array<MeasureInput>& inputs,
                                        const Array<BuildResult>& build_results, int verbose) {
  if (const auto* f = runtime::Registry::Get("auto_scheduler.rpc_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, key, host, port, priority, n_parallel, timeout, number, repeat,
             min_repeat_ms, cooldown_interval, enable_cpu_cache_flush, verbose);
    return results;
  } else {
    LOG(FATAL) << "auto_scheduler.rpc_runner.run is not registered. "
               << "This is a function registered in Python, "
               << "make sure the TVM Python runtime has been loaded successfully.";
  }
  return Array<MeasureResult>();
}

/********** MeasureCallback **********/
PythonBasedMeasureCallback::PythonBasedMeasureCallback(PackedFunc callback_func) {
  auto node = make_object<PythonBasedMeasureCallbackNode>();
  node->callback_func = std::move(callback_func);
  data_ = std::move(node);
}

void PythonBasedMeasureCallbackNode::Callback(const SearchPolicy& policy,

                                              // <bojian/DietCode>
                                              const ProgramMeasurer& measurer,

                                              const Array<MeasureInput>& inputs,
                                              const Array<MeasureResult>& results) {
  if (auto* sketch_policy = static_cast<SketchPolicyNode*>(policy.operator->())) {
    callback_func(GetRef<SketchPolicy>(sketch_policy), inputs, results);
  } else if (auto* empty_policy = static_cast<EmptyPolicyNode*>(policy.operator->())) {
    callback_func(GetRef<EmptyPolicy>(empty_policy), inputs, results);
  } else {
    LOG(FATAL) << "Unrecognized search policy type. Expect SketchPolicy or EmptyPolicy";
  }
}

/********** ProgramMeasurer **********/
ProgramMeasurer::ProgramMeasurer(ProgramBuilder builder, ProgramRunner runner,
                                 Optional<Array<MeasureCallback>> callbacks, int verbose,
                                 int max_continuous_error) {
  auto node = make_object<ProgramMeasurerNode>();
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->callbacks = std::move(callbacks);
  node->verbose = verbose;
  node->max_continuous_error = max_continuous_error < 0
                                   ? ProgramMeasurerNode::DEFAULT_MAX_CONTINUOUS_ERROR
                                   : max_continuous_error;
  data_ = std::move(node);
}

void ProgramMeasurerNode::Reset() {
  ct = error_ct = 0;
  
  // <bojian/DietCode>
  // best_flops.clear();
  // best_state.clear();
  best_score.clear();
  best_states.clear();
  best_state_flops.clear();
  best_inst_flops.clear();
  best_inst_disp_map.clear();
  
  best_ct.clear();
  has_valid.clear();
}

// <bojian/DietCode>
extern bool enable_verbose_logging;

Array<MeasureResult> ProgramMeasurerNode::Measure(const SearchTask& task,
                                                  const SearchPolicy& policy,
                                                  const Array<MeasureInput>& inputs,
                                                  int batch_size) {
  auto t_begin = std::chrono::high_resolution_clock::now();

  Array<MeasureResult> results;
  results.reserve(inputs.size());

  // <bojian/DietCode>
  // make a copy of the current best states and their corresponding flops
  std::vector<State> candidate_states;
  std::vector<float> candidate_flops;

  if (IsDynTask(task)) {
    // copy the best states and flops out as candidates
    candidate_states = std::move(best_states[task->workload_key]);
    candidate_flops  = std::move(best_state_flops[task->workload_key]);
  }

  if (batch_size == -1) {
    // set default batch size
    batch_size = builder->n_parallel * 2;
  }

  int old_verbosity = verbose;

  StdCout(verbose) << "Get " << inputs.size() << " programs to measure:" << std::endl;

  for (size_t i = 0; i < inputs.size(); i += batch_size) {
    Array<MeasureInput> input_batch(inputs.begin() + i,
                                    inputs.begin() + std::min(i + batch_size, inputs.size()));
    Array<MeasureResult> result_batch;

    // build and run
    SilentMeasure(task, input_batch, &result_batch);

    // update current best state according to the new measure result
    for (size_t j = 0; j < input_batch.size(); ++j) {
      const String& workload_key = input_batch[j]->task->workload_key;

      // <bojian/DietCode>
      // double flops
      Array<IntImm> cherry_picked_wkl_inst;
      double flop_ct, flops;
      float adaption_penalty;
      // Array<Array<PrimExpr>> factorization_scheme =
      //     input_batch[j]->state.GetFactorizationScheme();
      Array<Array<Optional<Integer>>> split_factors =
          input_batch[j]->state.GetSplitFactors();

      if (result_batch[j]->error_no == 0) {

        // <bojian/DietCode> Estimate the FLOPs for synthetic workloads.
        if (IsDynTask(task)) {
          // enable_verbose_logging = true;
          std::tie(cherry_picked_wkl_inst, flop_ct, adaption_penalty) =
              // GetSyntheticWorkloadFlopCtFromState(
              //   task, input_batch[j]->state);
              task->compute_dag.CherryPickWorkloadInstance(
                input_batch[j]->state, task);
          // enable_verbose_logging = false;
        } else {
          flop_ct = task->compute_dag->flop_ct;
          adaption_penalty = 1.;
        }
        // make sure that the value for FLOPS is well defined
        CHECK(flop_ct != -1);

        flops = // <bojian/DietCode>
                // task->compute_dag->flop_ct
                flop_ct / adaption_penalty
                  / FloatArrayMean(result_batch[j]->costs);

        LOG(INFO) << "Successfully completed the measurement on state w/ "
                     "split factors="
                       << OptionalMatrixToString(split_factors) << ", "
                     "costs=" << result_batch[j]->costs << "->"
                     "avg_cost=" << FloatArrayMean(result_batch[j]->costs) << ", "
                     "flop_ct=" << flop_ct << " => "
                     "flops=" << flop_ct / FloatArrayMean(result_batch[j]->costs) << "->" << flops;

        error_ct = 0;
        has_valid.insert(workload_key);
      } else {

        // <bojian/DietCode>
        // LOG(WARNING) << "Error encountered on state "
        //              << input_batch[j]->state << " w/ split factors="
        //              << OptionalMatrixToString(split_factors) << ", "
        //              << "error_msg=" << result_batch[j]->error_msg;

        flops = 0.0;
        error_ct++;
      }

      // <bojian/DietCode>
      if (IsDynTask(task)) {
        // record the measured throughputs
        candidate_states.push_back(input_batch[j]->state);
        candidate_flops .push_back(flops);

      } else {
        if (flops > best_score[workload_key]) {
          best_score[workload_key] = flops;
          
          // <bojian/DietCode>
          // best_state[workload_key] = input_batch[j]->state;
          best_states[workload_key] = {input_batch[j]->state};
          
          best_ct[workload_key] = ct;
        }
      }  // if (IsDynTask(task))

      ct++;
      StdCout(verbose, 2) << std::fixed << std::setprecision(2) << Chars('=', 50) << "\n"
                          << "No: " << ct << "\tGFLOPS: " << flops / 1e9 << " / "
                          
                          // <bojian/DietCode>
                          // << best_flops[workload_key] / 1e9 
                          << best_score[workload_key] / 1e9
                          
                          << "\tresults: " << result_batch[j]
                          << "\n"
                          << Chars('=', 50) << "\n"
                          << input_batch[j]->state << "\n";
    }  // for (j ∈ range(input_batch.size()))

    // Call callback functions
    if (callbacks
        
        // <bojian/DietCode>
        && !IsDynTask(policy->search_task)
    
        ) {

      // <bojian/DietCode>
      LOG(INFO) << "Invoking Callback method for a static search task";


      for (const auto& callback : callbacks.value()) {
        callback->Callback(policy,
                           
                           // <bojian/DietCode>
                           GetRef<ProgramMeasurer>(this),

                           input_batch, result_batch);
      }
    }

    // Store result batch
    for (auto& res : result_batch) {
      results.push_back(res);
    }

    if (error_ct > max_continuous_error) {
      LOG(WARNING) << "Too many errors happened during tuning. Switching to debug mode."
                   << std::endl;
      verbose = 2;
    } else {
      verbose = old_verbosity;
    }
  }

  // <bojian/DietCode>
  if (IsDynTask(task)) {

    // std::vector<std::string> candidate_states_str_repr;
    // std::ostringstream strout;
    // for (const State & state : candidate_states) {
    //   strout << "  " << OptionalMatrixToString(state.GetSplitFactors())
    //          << std::endl;
    //   candidate_states_str_repr.push_back(strout.str());
    //   strout.str("");
    //   strout.clear();
    // }
    // LOG(INFO) << "candidate_states="
    //           << ArrayToString(candidate_states_str_repr);
    // LOG(INFO) << "candidate_flops=" << ArrayToString(candidate_flops);

    // calculate the adapted score of each candidate state
    float occupancy_penalty, padding_penalty;
    // [num_insts x num_states]
    std::vector<float> adapted_candidate_flops(
        task->wkl_insts.size() * candidate_states.size());

    for (size_t state_id = 0; state_id < candidate_states.size(); ++state_id) {
      for (size_t inst_id = 0; inst_id < task->wkl_insts.size(); ++inst_id) {
        AdaptStateToWorkload(task, candidate_states[state_id],
                             task->wkl_insts[inst_id],
                             candidate_flops[state_id],
                             &occupancy_penalty, &padding_penalty,
                             &adapted_candidate_flops[
                               inst_id * candidate_states.size() + state_id]
                             );
      }
    }  // for (state_id ∈ candidate_states.size())

    // strout.str("");
    // strout.clear();
    // strout << "[";
    // for (size_t i = 0; i < adapted_candidate_flops.size(); ++i) {
    //   if (i % candidate_states.size() == 0) {
    //     strout << task->wkl_insts[i / candidate_states.size()] << ": ";
    //   }
    //   strout << adapted_candidate_flops[i] << ", ";
    //   if ((i + 1) % candidate_states.size() == 0) {
    //     strout << std::endl;
    //   }
    // }
    // strout << "]";
    // LOG(INFO) << "adapted_candidate_flops=" << strout.str();

    // Top-K Dispatch
    TopKDispatcher dispatcher;
    // enable_verbose_logging = true;
    std::unordered_map<size_t, size_t> raw_wkl_inst_id_disp_map =
        dispatcher.dispatch(adapted_candidate_flops, candidate_states.size());
    // enable_verbose_logging = false;
    // record the selected candidate states

    std::unordered_map<size_t, size_t> inst_id_disp_map;
    std::vector<State> selected_candidate_states;
    std::vector<float> selected_candidate_flops;
    std::vector<float> inst_predicted_flops;

    std::tie(inst_id_disp_map,
             selected_candidate_states,
             selected_candidate_flops,
             inst_predicted_flops) =
        dispatcher.MapWklInstsToStates(raw_wkl_inst_id_disp_map,
                                       candidate_states,
                                       candidate_flops,
                                       task->wkl_insts,
                                       adapted_candidate_flops);

    std::vector<std::string> selected_candidate_str_repr;
    std::ostringstream strout;
    strout.str("");
    strout.clear();
    for (const State& state : selected_candidate_states) {
      strout << "  " << OptionalMatrixToString(state.GetSplitFactors())
             << std::endl;
      selected_candidate_str_repr.push_back(strout.str());
      strout.str("");
      strout.clear();
    }
    Map<Array<IntImm>, Integer> inst_disp_map;
    for (const std::pair<size_t, size_t>& inst_state_pair : inst_id_disp_map) {
      inst_disp_map.Set(task->wkl_insts[inst_state_pair.first],
                        Integer(inst_state_pair.second));
    }

    // make a copy of the previous predicted FLOPS per instance
    std::vector<float> prev_inst_flops = std::move(best_inst_flops[task->workload_key]);

    LOG(INFO) << "best_states=" << ArrayToString(selected_candidate_str_repr);
    LOG(INFO) << "best_state_flops=" << ArrayToString(selected_candidate_flops);
    LOG(INFO) << "best_inst_disp_map=" << MapToString(inst_disp_map);
    LOG(INFO) << "best_inst_flops=" << ArrayToString(inst_predicted_flops);

    // inspect the predicted FLOPS per instance, and make sure that performance
    // degradation does not happen
    if (!prev_inst_flops.empty()) {
      for (size_t i = 0; i < task->wkl_insts.size(); ++i) {
        if (prev_inst_flops[i] > inst_predicted_flops[i]) {
          LOG(FATAL) << "Predicted FLOPS on inst="
                     << task->wkl_insts[i] << " dropped from "
                     << prev_inst_flops[i] << "=>" << inst_predicted_flops[i];
        }
      }
    }  // if (!prev_inst_flops.empty())

    best_states[task->workload_key] = std::move(selected_candidate_states);
    best_state_flops[task->workload_key] = std::move(selected_candidate_flops);
    best_inst_disp_map[task->workload_key] = std::move(inst_id_disp_map);
    best_inst_flops[task->workload_key] = std::move(inst_predicted_flops);

    if (callbacks) {
      
      LOG(INFO) << "Invoking Callback method for a dynamic search task";

      for (const auto& callback : callbacks.value()) {
        callback->Callback(policy, GetRef<ProgramMeasurer>(this), {}, {});
      }
    }

  }  // IsDynTask(task)

  PrintTimeElapsed(t_begin, "measurement", verbose);

  return results;
}

void ProgramMeasurerNode::SilentMeasure(const SearchTask& task, const Array<MeasureInput>& inputs,
                                        Array<MeasureResult>* results) {
  results->clear();
  results->reserve(inputs.size());

  // Call builder and runner
  Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
  Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);

  // Store result batch
  for (auto& res : result_batch) {
    results->push_back(res);
  }
}

/********** Printing functions **********/
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MeasureInputNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "MeasureInput()";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MeasureResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MeasureResultNode*>(ref.get());
      if (node->error_no == static_cast<int>(MeasureErrorNO::kNoError)) {
        p->stream << "MeasureResult(cost:[";
        auto old_config = p->stream.precision(4);
        for (size_t i = 0; i < node->costs.size(); ++i) {
          auto pf = node->costs[i].as<FloatImmNode>();
          ICHECK(pf != nullptr);
          p->stream << pf->value;
          if (i != node->costs.size() - 1) {
            p->stream << ",";
          }
        }
        p->stream.precision(old_config);
        p->stream << "], ";
        p->stream << "error_no:" << 0 << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      } else {
        p->stream << "MeasureResult("
                  << "error_type:" << ErrorNoToStr[node->error_no] << ", "
                  << "error_msg:" << node->error_msg << ", "
                  << "all_cost:" << node->all_cost << ", "
                  << "Tstamp:" << node->timestamp << ")";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BuildResultNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BuildResultNode*>(ref.get());
      p->stream << "BuildResult(" << node->filename << ", " << node->error_no << ", "
                << node->time_cost << ")";
    });

/********** Measure interface API for ffi **********/
TVM_REGISTER_GLOBAL("auto_scheduler.MeasureInput").set_body_typed([](SearchTask task, State state) {
  return MeasureInput(task, state);
});

TVM_REGISTER_GLOBAL("auto_scheduler.BuildResult")
    .set_body_typed([](String filename, Array<te::Tensor> args, int error_no, String error_msg,
                       double time_cost) {
      return BuildResult(filename, args, error_no, error_msg, time_cost);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.MeasureResult")
    .set_body_typed([](Array<PrimExpr> costs, int error_no, String error_msg, double all_cost,
                       double timestamp) {
      return MeasureResult(costs, error_no, error_msg, all_cost, timestamp);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.PythonBasedMeasureCallback")
    .set_body_typed([](PackedFunc callback_func) {
      return PythonBasedMeasureCallback(callback_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ProgramMeasurer")
    .set_body_typed([](ProgramBuilder builder, ProgramRunner runner,
                       Array<MeasureCallback> callbacks, int verbose, int max_continuous_error) {
      return ProgramMeasurer(builder, runner, callbacks, verbose, max_continuous_error);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ProgramBuilderBuild")
    .set_body_typed([](const ProgramBuilder& builder, const Array<MeasureInput>& inputs,
                       int verbose) { return builder->Build(inputs, verbose); });

TVM_REGISTER_GLOBAL("auto_scheduler.ProgramRunnerRun")
    .set_body_typed([](const ProgramRunner& runner, const Array<MeasureInput>& inputs,
                       const Array<BuildResult>& build_results,
                       int verbose) { return runner->Run(inputs, build_results, verbose); });

TVM_REGISTER_GLOBAL("auto_scheduler.LocalBuilder")
    .set_body_typed([](int timeout, int n_parallel, const String& build_func) {
      return LocalBuilder(timeout, n_parallel, build_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.LocalRunner")
    .set_body_typed([](int timeout, int number, int repeat, int min_repeat_ms,
                       double cooldown_interval, bool enable_cpu_cache_flush) {
      return LocalRunner(timeout, number, repeat, min_repeat_ms, cooldown_interval,
                         enable_cpu_cache_flush);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.RPCRunner")
    .set_body_typed([](const String& key, const String& host, int port, int priority,
                       int n_parallel, int timeout, int number, int repeat, int min_repeat_ms,
                       double cooldown_interval, bool enable_cpu_cache_flush) {
      return RPCRunner(key, host, port, priority, n_parallel, timeout, number, repeat,
                       min_repeat_ms, cooldown_interval, enable_cpu_cache_flush);
    });

}  // namespace auto_scheduler
}  // namespace tvm
