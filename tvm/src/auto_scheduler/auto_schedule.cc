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
 * \file auto_scheduler/auto_schedule.cc
 * \brief The user interface and tuning options of the TVM auto-scheduler.
 */

#include <tvm/auto_scheduler/auto_schedule.h>

// <bojian/DietCode>
#include <tvm/auto_scheduler/dietcode.h>

#include <tvm/runtime/registry.h>

#include "utils.h"
#include "search_policy/utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(TuningOptionsNode);

TuningOptions::TuningOptions(int num_measure_trials, int early_stopping, int num_measures_per_round,
                             int verbose, ProgramBuilder builder, ProgramRunner runner,
                             Optional<Array<MeasureCallback>> measure_callbacks) {
  auto node = make_object<TuningOptionsNode>();
  node->num_measure_trials = num_measure_trials;
  node->early_stopping = early_stopping;
  node->num_measures_per_round = num_measures_per_round;
  node->verbose = verbose;
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->measure_callbacks = std::move(measure_callbacks);
  data_ = std::move(node);
}

// <bojian/DietCode>
// std::pair<te::Schedule, Array<te::Tensor>>
ObjectRef  // DynWklDispatcher in the case of dynamic workload,
           // (state, schedule, tensors) otherwise
AutoSchedule(SearchPolicy search_policy, TuningOptions tuning_options) {
  // Create a ProgramMeasurer to handle the schedule build and performance measure
  ProgramMeasurer measurer =
      ProgramMeasurer(tuning_options->builder, tuning_options->runner,
                      tuning_options->measure_callbacks, tuning_options->verbose);
  // Search for the best schedule
  std::vector<State> states;
  std::unordered_map<size_t, size_t> inst_disp_map;
  std::tie(states, inst_disp_map) =
      search_policy->Search(tuning_options->num_measure_trials,
                            tuning_options->early_stopping,
                            tuning_options->num_measures_per_round,
                            measurer);
                            // <bojian/DietCode>
                            // [0];

  // <bojian/DietCode>
  if (IsDynTask(search_policy->search_task)) {
    return DynWklDispatcher(search_policy->search_task,
                            std::move(states), std::move(inst_disp_map));
  } else {
    CHECK(states.size() == 1);
    State state = Downcast<State>(states[0]);
    if (state.defined()) {
      std::pair<te::Schedule, Array<te::Tensor>> sch_and_tensors =
          search_policy->search_task->compute_dag.ApplySteps(
            state->transform_steps);
      return Array<ObjectRef>{
               state, sch_and_tensors.first, sch_and_tensors.second
             };
    } else {
      StdCout(tuning_options->verbose)
          << "No valid state found in this search round. Check if it has traversed all of the "
          << "search space." << std::endl;
      // Return the default schedule
      return Array<ObjectRef>{
               search_policy->search_task->compute_dag->init_state,
               te::Schedule(search_policy->search_task->compute_dag->ops),
               search_policy->search_task->compute_dag->tensors
             };
    }
  }  // if (IsDynTask(search_policy->search_task))
}

TVM_REGISTER_GLOBAL("auto_scheduler.TuningOptions")
    .set_body_typed([](int num_measure_trials, int early_stopping, int num_measures_per_round,
                       int verbose, ProgramBuilder builder, ProgramRunner runner,
                       Optional<Array<MeasureCallback>> measure_callbacks) {
      return TuningOptions(num_measure_trials, early_stopping, num_measures_per_round, verbose,
                           builder, runner, measure_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.AutoSchedule")
    .set_body_typed([](SearchPolicy search_policy, TuningOptions tuning_options) {
      // <bojian/DietCode> Changed the AutoSchedule interface.
      // te::Schedule sch;
      // Array<te::Tensor> return_tensors;
      // std::tie(sch, return_tensors) = AutoSchedule(search_policy, tuning_options);
      // return Array<ObjectRef>{sch, return_tensors};
      return AutoSchedule(search_policy, tuning_options);
    });


}  // namespace auto_scheduler
}  // namespace tvm
