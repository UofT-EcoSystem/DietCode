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
 * \file auto_scheduler/search_policy/utils.cc
 * \brief Common utilities
 */

#include "utils.h"

#include <algorithm>

namespace tvm {
namespace auto_scheduler {

Array<Integer> GetSpatialSplitStepIds(const State& s, int stage_id) {
  const auto& stage = s->stages[stage_id];
  const auto& pop = s->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  size_t reduce_count = 0;
  for (const auto axis : pop->reduce_axis) {
    if (!no_split_at_inner_name_set.count(axis->var->name_hint)) {
      reduce_count++;
    }
  }

  Array<Integer> spatial_split_step_ids;
  for (int i = s->transform_steps.size() - 1; i >= 0; --i) {
    if (IsStageNumberChangingStep(s->transform_steps[i])) {
      if (stage_id > s->transform_steps[i]->stage_id) {
        stage_id--;
      }
    } else if (auto ps = s->transform_steps[i].as<SplitStepNode>()) {
      if (stage_id == ps->stage_id) {
        // Assume SplitStep on reduction axes are always after SplitStep on spatial axes.
        if (reduce_count) {
          reduce_count--;
        } else {
          spatial_split_step_ids.push_back(i);
        }
      }
    }
  }

  return spatial_split_step_ids;
}

std::vector<std::pair<int, int>> GetComputeLocationCandidates(const SearchTask& task,
                                                              const State& state, int stage_id) {
  int target_stage_id = GetSingleConsumerId(task, state, stage_id);
  if (target_stage_id < 0) {
    return {};
  }
  const Stage& target_stage = state->stages[target_stage_id];

  std::vector<std::pair<int, int>> candidates;
  bool target_compute_at_other = target_stage->compute_at == ComputeAtKind::kIter;
  bool target_is_tiled = IsTiled(target_stage);

  bool visited_reduce = false;
  // Enumerate compute_at location at target_stage
  // TODO(merrymercy): More analysis here to make smarter choices
  for (size_t i = 0; i < target_stage->iters.size(); ++i) {
    const Iterator& target_iter = target_stage->iters[i];
    if (target_iter->iter_kind == IteratorKind::kReduction) {
      visited_reduce = true;
      if (!target_is_tiled) {  // Do not go into reduce iter
        break;
      }
    } else if (target_iter->iter_kind == IteratorKind::kSpatial) {
      if (visited_reduce) {  // Do not go into inner tile
        break;
      }
    }

    if (target_iter->annotation == IteratorAnnotation::kUnroll) {
      // Do not go into the unroll region of const tensor indices
      break;
    }

    if (GetExtent(target_iter) == 1) {
      // Skip iterators with length of 1
      continue;
    }
    if (target_compute_at_other && target_iter->iter_kind == IteratorKind::kSpatial &&
        StrEndsWith(target_iter->name, ".0")) {
      // Skip the first level iterators if target stage compute_at another stage
      // In this case, the lengths of first level iterators are always one
      continue;
    }
    candidates.emplace_back(target_stage_id, i);

    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(target_stage_id, i))) {
      break;
    }
  }

  // if the target_stage is already compute_at another stage X, try also compute_at X
  // We call stage X as `target_target_stage`
  if (target_compute_at_other) {
    int target_target_stage_id;
    target_target_stage_id = state->attach_map->stage_to_attach_iter.at(target_stage_id).first;
    const Stage& target_target_stage = state->stages[target_target_stage_id];

    for (size_t i = 0; i < target_target_stage->iters.size(); ++i) {
      const Iterator& target_target_iter = target_target_stage->iters[i];
      if (target_target_iter->iter_kind == IteratorKind::kReduction ||
          state->attach_map->iter_to_attached_stages.count(
              std::make_pair(target_target_stage_id, i))) {
        break;
      }

      if (target_target_iter->annotation == IteratorAnnotation::kUnroll) {
        // Do not go into the unroll region of const tensor indices
        break;
      }

      if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }

      candidates.emplace_back(target_target_stage_id, i);
    }
  }

  return candidates;
}

State DoMultiLevelTiling(const State& state, int stage_id, const std::string& format,
                         std::vector<int>* spatial_split_step_ids
                         
                         // <bojian/DietCode>
                         // , bool simplify_tiling_structure
                         
                         ) {
  // Temporal object to be used if the input pointer is nullptr
  std::vector<int> temp_split_step_ids;
  if (spatial_split_step_ids == nullptr) {
    spatial_split_step_ids = &temp_split_step_ids;
  }
  spatial_split_step_ids->clear();

  std::vector<std::vector<Iterator>> space_levels;
  std::vector<std::vector<Iterator>> reduce_levels;
  std::vector<Iterator> space_outer, space_inner, reduce_outer, reduce_inner;

  size_t n_space =
      std::count(format.begin(), format.end(), 's') + std::count(format.begin(), format.end(), 'S');
  size_t n_reduce =
      std::count(format.begin(), format.end(), 'r') + std::count(format.begin(), format.end(), 'R');
  if (n_space + n_reduce != format.size()) {
    LOG(FATAL) << "Invalid multi-level tiling format: " << format;
  }
  space_levels.resize(n_space);
  reduce_levels.resize(n_reduce);

  State tmp_s = state;
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();

  auto sr_levels = [&](int size, const Iterator& iter, std::vector<std::vector<Iterator>>& levels) {
    ICHECK_GE(size, 1);
    if (size == 1) {
      levels[0].push_back(iter);
    } else {
      Array<Iterator> split_res =
          tmp_s.split(stage_id, iter, Array<Optional<Integer>>(size - 1, NullOpt));
      for (int i = 0; i < size; i++) {
        levels[i].push_back(split_res[i]);
      }
      if (iter->iter_kind == IteratorKind::kSpatial) {
        spatial_split_step_ids->push_back(tmp_s->transform_steps.size() - 1);
      }
    }
  };

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (!no_split_at_inner_name_set.count(iter->name)) {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        sr_levels(n_space, iter, space_levels);
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        sr_levels(n_reduce, iter, reduce_levels);
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    } else {
      if (iter->iter_kind == IteratorKind::kSpatial) {
        space_inner.push_back(iter);
      } else if (iter->iter_kind == IteratorKind::kReduction) {
        reduce_inner.push_back(iter);
      } else {
        LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
      }
    }
  }

  auto fill_levels = [&](std::vector<Iterator>& levels_iter, std::vector<Iterator>& fill) {
    if (!fill.empty()) {
      levels_iter.insert(levels_iter.begin(), std::make_move_iterator(fill.begin()),
                         std::make_move_iterator(fill.end()));
    }
  };
  if (!space_levels.empty()) {
    fill_levels(space_levels.front(), space_outer);
    fill_levels(space_levels.back(), space_inner);
  }
  if (!reduce_levels.empty()) {
    fill_levels(reduce_levels.front(), reduce_outer);
    fill_levels(reduce_levels.back(), reduce_inner);
  }

  Array<Iterator> order;
  int space_ct = 0, reduce_ct = 0;
  for (const auto c : format) {
    if (c == 's' || c == 'S') {
      order.insert(order.end(), std::make_move_iterator(space_levels[space_ct].begin()),
                   std::make_move_iterator(space_levels[space_ct].end()));
      space_ct++;
    } else if (c == 'r' || c == 'R') {
      order.insert(order.end(), std::make_move_iterator(reduce_levels[reduce_ct].begin()),
                   std::make_move_iterator(reduce_levels[reduce_ct].end()));
      reduce_ct++;
    } else {
      LOG(FATAL) << "Invalid multi level tiling format: " << format;
    }
  }

  tmp_s.reorder(stage_id, order);
  return tmp_s;
}

State FollowTiling(const State& state, int stage_id, const std::vector<int>& split_step_ids,
                   int n_split) {
  if (n_split < 1 || n_split > 3) {
    LOG(FATAL) << "Invalid split parts, currently only support 1, 2 and 3";
  }
  // Apply up to three-level tiling structure:  space_L0, space_L1, space_L2
  std::vector<Iterator> space_0, space_1, space_2, space_3, tmp_order;
  Array<Iterator> split_res;

  auto pop = state->stages[stage_id]->op.as<te::ComputeOpNode>();
  ICHECK(pop != nullptr);
  const Stage& stage = state->stages[stage_id];
  const std::set<std::string>& no_split_at_inner_name_set =
      stage->op->attrs.count(SearchPolicyKey::no_split_at_inner)
          ? GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::no_split_at_inner)
          : std::set<std::string>();
  int no_split_at_inner_name_in_stage_cnt = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    no_split_at_inner_name_in_stage_cnt += no_split_at_inner_name_set.count(iter->name);
  }

  ICHECK_EQ(state->stages[stage_id]->iters.size() - no_split_at_inner_name_in_stage_cnt,
            split_step_ids.size());

  State tmp_s = state;
  int ct = 0;
  for (const auto& iter : state->stages[stage_id]->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      // For spatial iterator, split it into multi iterators
      if (!no_split_at_inner_name_set.count(iter->name)) {
        IteratorAnnotation ann_type = iter->annotation;
        split_res = tmp_s.follow_split(stage_id, iter, split_step_ids[ct], n_split);
        // Restore annotation. Move unroll and vectorize to inner, move parallel
        // to outer
        switch (ann_type) {
          case IteratorAnnotation::kUnroll:
            split_res.Set(n_split, tmp_s.unroll(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kVectorize:
            split_res.Set(n_split, tmp_s.vectorize(stage_id, split_res[n_split]));
            break;
          case IteratorAnnotation::kParallel:
            split_res.Set(0, tmp_s.parallel(stage_id, split_res[0]));
            break;
          default:
            break;
        }

        space_0.push_back(split_res[0]);
        space_1.push_back(split_res[1]);
        if (n_split >= 2) {
          space_2.push_back(split_res[2]);
          if (n_split == 3) {
            space_3.push_back(split_res[3]);
          }
        }
        ct++;
      } else {
        if (no_split_at_inner_name_set.count(iter->name)) {
          if (n_split == 1) {
            space_1.push_back(iter);
          } else if (n_split == 2) {
            space_2.push_back(iter);
          } else {
            ICHECK_EQ(n_split, 3);
            space_3.push_back(iter);
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid iter type: " << int(iter->iter_kind);
    }
  }

  if (n_split == 3) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2, &space_3);
  } else if (n_split == 2) {
    ConcatenateMove(&tmp_order, &space_0, &space_1, &space_2);
  } else {
    ConcatenateMove(&tmp_order, &space_0, &space_1);
  }
  tmp_s.reorder(stage_id, tmp_order);
  return tmp_s;
}

// Return whether a state has nested parallel, which is invalid on CPUs
bool HasNestedParallel(const State& state) {
  std::function<void(int stage_id, size_t*)> count_parallel_ct;

  count_parallel_ct = [&state, &count_parallel_ct](int stage_id, size_t* parallel_ct) {
    const Stage& stage = state->stages[stage_id];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      return;
    }

    for (size_t i = 0; i < stage->iters.size(); ++i) {
      if (stage->iters[i]->annotation == IteratorAnnotation::kParallel) {
        (*parallel_ct)++;
      }

      IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          count_parallel_ct(attach_stage_id, parallel_ct);
        }
      }
    }
  };

  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    size_t parallel_ct = 0;

    if (state->stages[stage_id]->compute_at == ComputeAtKind::kRoot) {
      count_parallel_ct(stage_id, &parallel_ct);
      if (parallel_ct >= 2) {
        return true;
      }
    }
  }

  return false;
}

void PruneInvalidState(const SearchTask& task, Array<State>* states) {
  size_t pt = 0;
  for (size_t i = 0; i < states->size(); ++i) {
    if (!(*states)[i].defined()) {
      continue;
    }
    if (!IsGPUTask(task) && HasNestedParallel((*states)[i])) {
      continue;
    }

    if (i != pt) {
      states->Set(pt, (*states)[i]);
    }
    pt++;
  }

  if (pt == 0) {
    LOG(FATAL) << "Internal error: All states are invalid.";
  } else {
    states->resize(pt);
  }
}

/********** SplitFactorizationMemo **********/

extern bool is_sample_init_population_1st_iter;
extern bool enable_verbose_logging;

const Array<Array<Integer>>& SplitFactorizationMemo::GetFactorizationSchemes(
    int extent, int n_lengths
    // , int max_innermost_factor
    ) {
  QueryKey key = // std::make_tuple(extent, n_lengths, max_innermost_factor);
                 std::make_pair(extent, n_lengths);
  const auto& it = memory_.find(key);
  if (it != memory_.end()) {
    return it->second;
  }
  // if (!is_sample_init_population_1st_iter) {
  //   LOG(FATAL) << "(extent=" << extent << ", n_lengths=" << n_lengths <<") has "
  //                 "not been found in SplitFactorizationMemo";
  // }

  tmp_stack_ = Array<Integer>(n_lengths, Integer());
  results_ = &memory_[key];
  n_lengths_ = n_lengths;

  DfsEnumerate(0, extent
               // , max_innermost_factor
               );

  return *results_;
}

void SplitFactorizationMemo::DfsEnumerate(int now, int remaining_length
                                          // , int max_innermost_factor
                                          ) {
  if (now == n_lengths_) {
    if (tmp_stack_.back().as<IntImmNode>()->value <=
        // max_innermost_factor
        max_innermost_factor_
        // in case the maximum innermost factor is not defined
        || max_innermost_factor_ == 0
        ) {
      results_->push_back(tmp_stack_);
    }
  } else {
    for (const auto& f : GetFactors(remaining_length)) {
      tmp_stack_.Set(now, Integer(f));
      DfsEnumerate(now + 1, remaining_length / f
                   // , max_innermost_factor
                   );
    }
  }
}

const std::vector<int>& SplitFactorizationMemo::GetFactors(int n) {
  auto it = factor_memory_.find(n);
  if (it != factor_memory_.end()) {
    return it->second;
  }

  std::vector<int>& res = factor_memory_[n];
  int step = n % 2 == 0 ? 1 : 2;
  for (size_t i = 1; i < static_cast<size_t>(std::sqrt(n)) + 1; i += step) {
    if (n % i == 0) {
      res.push_back(i);
      if (n / i != i) {
        res.push_back(n / i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}


// <bojian/DietCode>

// std::vector<SplitStepInfo> FactorizationScheme::split_steps_info;
// bool FactorizationScheme::simplify_sketch;
// size_t FactorizationScheme::last_spatial_iter_id;
// std::vector<std::vector<int>> FactorizationScheme::factor_indices_to_incr;

// FactorizationScheme::FactorizationScheme(
//     const std::vector<SplitStepInfo>& split_steps_info,
//     const bool simplify_sketch, const bool init_static_members) {
//   for (size_t i = 0; i < split_steps_info.size(); ++i) {
//     split_factors.push_back(
//         std::vector<int>(GetSplitLengths(split_steps_info[i].is_spatial), 1));
//   }
//   if (init_static_members) {
//     FactorizationScheme::split_steps_info = split_steps_info;
//     FactorizationScheme::simplify_sketch = simplify_sketch;
//     last_spatial_iter_id = -1;
//     for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
//       if (split_steps_info[iter_id].is_spatial) {
//         last_spatial_iter_id = iter_id;
//       }
//     }
//     if (simplify_sketch) {
//       for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
//         if (split_steps_info[iter_id].is_spatial) {
//           if (iter_id == last_spatial_iter_id) {
//             factor_indices_to_incr.push_back({0, 1});
//           } else {
//             factor_indices_to_incr.push_back({1, 3});
//           }
//         } else {
//           factor_indices_to_incr.push_back({1});
//         }
//       }  // for (iter_id ∈ split_steps_info.size())
//     }    // if (simplify_sketch)
//   } else {
//     CHECK(FactorizationScheme::split_steps_info == split_steps_info);
//     CHECK(FactorizationScheme::simplify_sketch == simplify_sketch);
//   }
// }


constexpr size_t C_EXTENT_THRESHOLD = 16, C_MIN_NUM_FACTORS = 4;
constexpr size_t C_NUM_THREADS_PER_BLOCK_SIZE = 11;
constexpr size_t C_NUM_THREADS_PER_BLOCK[C_NUM_THREADS_PER_BLOCK_SIZE] =
    {32, 64, 96, 128, 256, 384, 512, 640, 768, 896, 1024};


static void shrink_below_max_extent(int& curr_split_factor,
                                    const size_t max_extent) {
  bool should_shrink_f = true;
  while (should_shrink_f) {
    if (curr_split_factor == 1) {
      break;
    }
    // const std::vector<int>& f_factors = memo.GetFactors(curr_split_factor);
    // CHECK(f_factors.size() >= 2);

    if (floor_div(curr_split_factor, 2) > static_cast<int>(max_extent * 1.1)) {
      if (enable_verbose_logging) {
        LOG(INFO) << "Shrinking orig_split_factor=" << curr_split_factor << " to "
                     "new_split_factor=" << curr_split_factor / 2;
      }
      curr_split_factor = floor_div(curr_split_factor, 2);
      should_shrink_f = true;
    } else {
      should_shrink_f = false;
    }
  }  // while (should_shrink_f)
}

std::vector<int> mutate_split_factors(const std::vector<int>& input_split_factors,
                                      const int max_innermost_extent,
                                      std::mt19937* const rng,
                                      SplitFactorizationMemo& memo) {
  std::vector<int> random_perm;
  RandomPermutation(input_split_factors.size(), &random_perm, rng);

  // try to divide a factor from one tile size and multiple it to another.
  for (size_t i = 0; i < random_perm.size(); ++i) {
    size_t src_idx = random_perm[i];
    int src_factor = input_split_factors[src_idx];
    if (src_factor <= 1) {
      continue;
    }

    // divide one factor from lengths[src_idx] and multiply it to lengths[dst_idx]
    size_t dst_idx = random_perm[(i + 1) % random_perm.size()];
    const std::vector<int>& factors = memo.GetFactors(src_factor);
    ICHECK_GE(factors.size(), 1);

    int divide_factor;
    if (dst_idx == input_split_factors.size() - 1) {
      // maintain the restriction of hardware_params.max_innermost_split_factor.
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * input_split_factors[dst_idx] <= max_innermost_extent) {
          break;
        }
      }
      if (max_factor_index == 0) {
        // failed on this dst_idx, try next one.
        continue;
      }

      divide_factor = factors[1 + (*rng)() % (max_factor_index)];
    } else {
      divide_factor = factors[1 + (*rng)() % (factors.size() - 1)];
    }

    // divide one factor from lengths[src_idx] and multiply it to lengths[dst_idx].
    std::vector<int> new_split_factors;
    for (size_t j = 0; j < input_split_factors.size(); ++j) {
      if (j == src_idx) {
        new_split_factors.push_back(Integer(input_split_factors[j] / divide_factor));
      } else if (j == dst_idx) {
        new_split_factors.push_back(Integer(input_split_factors[j] * divide_factor));
      } else {
        new_split_factors.push_back(Integer(input_split_factors[j]));
      }
    }
    return new_split_factors;
  }
  return input_split_factors;
}


void FactorizationScheme::RandomSample(const std::vector<SplitStepInfo>& split_steps_info,
                                       const HardwareParams& hardware_params,
                                       const size_t max_innermost_factor,
                                       std::mt19937* const rng,
                                       const bool do_mutation,
                                       const bool sample_perfect_tiles) {
  // ===========================================================================
  // 1. factor[1] (threadIdx.x)
  // ===========================================================================
  size_t num_threads_per_block = 0;
  SplitFactorizationMemo memo;

  if (!do_mutation) {
    // 1. Fix factor[1], only mutate the factor[0] or factor[3].
    size_t num_spatial_axes = 0;
    for (const SplitStepInfo& info : split_steps_info) {
      if (info.is_spatial) {
        ++num_spatial_axes;
      }
    }

    std::uniform_int_distribution<>
        num_threads_per_block_dist(0, C_NUM_THREADS_PER_BLOCK_SIZE - 1);
    num_threads_per_block = C_NUM_THREADS_PER_BLOCK[num_threads_per_block_dist(*rng)];

    // find all the possible factors of the number of threads per block
    if (enable_verbose_logging) {
      LOG(INFO) << "num_threads_per_block=" << num_threads_per_block;
      LOG(INFO) << "num_spatial_axes=" << num_spatial_axes;
    }
    const Array<Array<Integer>>& num_threads_factor_schemes =
        memo.GetFactorizationSchemes(num_threads_per_block, num_spatial_axes - 1);

    // if (enable_verbose_logging) {
    //   LOG(INFO) << "Sampled the factors out of num_threads_factor_schemes.size()="
    //             << num_threads_factor_schemes.size();
    // }
    CHECK(num_threads_factor_schemes.size() != 0);

    // avoid do-while loops, early filter out factorization schemes that are not
    // valid (i.e., have split factors greater than the maximum extent)
    Array<Array<Integer>> filtered_num_threads_factor_schemes;

    filtered_num_threads_factor_schemes.reserve(num_threads_factor_schemes.size());
    for (Array<Integer> factor_scheme : num_threads_factor_schemes) {
      int64_t factor_prod = 1;
      for (const Integer& factor : factor_scheme) {
        factor_prod *= factor;
      }
      factor_scheme.push_back(num_threads_per_block / factor_prod);
      if (enable_verbose_logging) {
        LOG(INFO) << "Factorization Scheme for Threads Per Block: "
                  << factor_scheme;
      }

      // check whether the factorization scheme is valid or not
      bool all_below_max_extents = true;
      for (size_t iter_id = 0, spatial_iter_id = 0;
           iter_id < split_steps_info.size(); ++iter_id) {
        if (split_steps_info[iter_id].is_spatial) {
          if (factor_scheme[spatial_iter_id]->value >
                static_cast<int>(
                  split_steps_info[iter_id].max_extent * 1.1
                )
              ) {
            all_below_max_extents = false;
            break;
          }
          ++spatial_iter_id;
        }
      }  // for (iter_id ∈ [0, split_steps_info.size()))
      if (!all_below_max_extents) {
        continue;
      }
      filtered_num_threads_factor_schemes.push_back(factor_scheme);
    }

    const Array<Integer> factor_scheme =
        RandomChooseAmong(filtered_num_threads_factor_schemes, rng);
    // bool all_below_max_extents;
    // Array<Integer> num_threads_factor_scheme;
    // do {
    //   all_below_max_extents = true;

    //   num_threads_factor_scheme = num_threads_factor_schemes[num_threads_factor_schemes_dist(*rng)];
    //   // if (enable_verbose_logging) {
    //   //   LOG(INFO) << "Factorization Scheme: " << ArrayToString(num_threads_factor_scheme);
    //   // }
    //   int64_t factor_prod = 1;
    //   for (const Integer& factor : num_threads_factor_scheme) {
    //     factor_prod *= factor;
    //   }
    //   num_threads_factor_scheme.push_back(num_threads_per_block / factor_prod);
    //   if (enable_verbose_logging) {
    //     LOG(INFO) << "Factorization Scheme: " << ArrayToString(num_threads_factor_scheme);
    //   }
    //   for (size_t iter_id = 0, spatial_iter_id = 0;
    //        iter_id < split_steps_info.size(); ++iter_id) {
    //     if (split_steps_info[iter_id].is_spatial) {
    //       if (num_threads_factor_scheme[spatial_iter_id]->value >
    //             split_steps_info[iter_id].max_extent) {
    //         all_below_max_extents = false;
    //       }
    //       ++spatial_iter_id;
    //     }
    //   }  // for (iter_id ∈ [0, split_steps_info.size()))
    // } while (!all_below_max_extents);
    // do the looping again and assign the factors
    for (size_t iter_id = 0, spatial_iter_id = 0;
         iter_id < split_steps_info.size(); ++iter_id) {
      if (split_steps_info[iter_id].is_spatial) {
        split_factors[iter_id][1] = factor_scheme[spatial_iter_id]->value;
        ++spatial_iter_id;
      }
    }
  } else {  // if (do_mutation)

    // LOG(INFO) << "num_threads_per_block=" << num_threads_per_block;

    for (size_t iter_id = 0, spatial_iter_id = 0;
         iter_id < split_steps_info.size(); ++iter_id) {
      if (split_steps_info[iter_id].is_spatial) {
        shrink_below_max_extent(split_factors[iter_id][1],
                                split_steps_info[iter_id].max_extent);
        ++spatial_iter_id;
      }
    }

    num_threads_per_block = 1;
    for (const std::vector<int>& split_factor : split_factors) {
      if (split_factor.size() == 2) {
        continue;
      }
      num_threads_per_block *= split_factor[1];
    }
    // LOG(INFO) << "num_threads_per_block=" << num_threads_per_block;
  }  // if (!do_mutation)

  size_t last_spatial_iter_id = -1;
  for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
    if (split_steps_info[iter_id].is_spatial) {
      last_spatial_iter_id = iter_id;
    }
  }
  CHECK(last_spatial_iter_id != -1UL);


  // ===========================================================================
  // factor[0] (vthread)
  // ===========================================================================
  // size_t reg_usage = num_threads_per_block, shmem_usage = 0;

  size_t iter_id_to_mutate = -1;
  if (do_mutation) {
    iter_id_to_mutate = (*rng)() % split_steps_info.size();
  }

  auto sample_factors = [&](std::function<bool(const size_t)>   fcontinue_predicate,
                            std::function<size_t(const size_t)> fmax_extent,
                            std::function<int&(const size_t)>   ffactor_to_assign,
                            std::function<size_t(const size_t)> fextent_to_factor) {
        // std::vector<size_t> max_extents;
        std::vector<size_t> factors_to_assign;
        for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {

          if (fcontinue_predicate(iter_id)) {
            continue;
          }

          const size_t max_extent =
              std::min(fmax_extent(iter_id), fextent_to_factor(iter_id));

          if (do_mutation) {
            if (iter_id != iter_id_to_mutate) {
              continue;
            }
            int& curr_inner_factor = ffactor_to_assign(iter_id);
            shrink_below_max_extent(curr_inner_factor, max_extent);
            if (split_steps_info[iter_id].is_spatial) {
              const int curr_nthreads_factor = split_factors[iter_id][1];
              int curr_outer_factor = floor_div(fextent_to_factor(iter_id),
                                                curr_inner_factor);
              CHECK(curr_outer_factor >= 1);
              std::vector<int> mutated_split_factors =
                  mutate_split_factors({curr_outer_factor, curr_nthreads_factor, curr_inner_factor},
                                       fmax_extent(iter_id), rng, memo);
              CHECK(mutated_split_factors.size() == 3);
              CHECK(mutated_split_factors[0] >= 1);
              CHECK(mutated_split_factors[1] >= 1);
              CHECK(mutated_split_factors[2] >= 1);

              // LOG(FATAL) << "orig_split_factors="
              //                 << ArrayToString(std::vector<int>{curr_outer_factor, curr_nthreads_factor,
              //                                                   curr_inner_factor}) << ", "
              //               "mutated_split_factors=" << ArrayToString(mutated_split_factors);

              split_factors[iter_id][1] = mutated_split_factors[1];
              ffactor_to_assign(iter_id) = mutated_split_factors[2];

            } else {
              int curr_outer_factor = floor_div(fextent_to_factor(iter_id),
                                                curr_inner_factor);
              CHECK(curr_outer_factor >= 1);

              std::vector<int> mutated_split_factors =
                  mutate_split_factors({curr_outer_factor, curr_inner_factor},
                                       fmax_extent(iter_id), rng, memo);
              CHECK(mutated_split_factors.size() == 2);
              CHECK(mutated_split_factors[0] >= 1);
              CHECK(mutated_split_factors[1] >= 1);

              // LOG(FATAL) << "orig_split_factors="
              //                 << ArrayToString(std::vector<int>{curr_outer_factor, curr_inner_factor}) << ", "
              //               "mutated_split_factors=" << ArrayToString(mutated_split_factors);


              ffactor_to_assign(iter_id) = mutated_split_factors[1];
            }
            // directly return after a mutation has been made
            return;
          }  // if (do_mutation)

          // sampling
          CHECK(max_extent > 0);
          size_t factor_to_assign = 0;

          if (sample_perfect_tiles) {
            // LOG(INFO) << "Sampling perfect tile size of "
            //           << extent_to_factor(iter_id) << " vs. max="
            //           << iter_max_extent;
            const size_t extent_to_factor = fextent_to_factor(iter_id);
            const std::vector<int>& extent_factors = memo.GetFactors(extent_to_factor);
            std::vector<int> filtered_extent_factors;
            filtered_extent_factors.reserve(extent_factors.size());

            for (const int factor : extent_factors) {
              if (static_cast<size_t>(factor) <= static_cast<int>(max_extent * 1.1)) {
                filtered_extent_factors.push_back(factor);
              }
            }

            if (!filtered_extent_factors.empty() &&
                (extent_to_factor <= C_EXTENT_THRESHOLD || 
                 (extent_to_factor > C_EXTENT_THRESHOLD && filtered_extent_factors.size() >= C_MIN_NUM_FACTORS))
                ) {
              factor_to_assign =
                  RandomChooseAmong(filtered_extent_factors, rng);
            } // else {
            //   LOG(WARNING) << "Sampling imperfect tile size. "
            //                   "filtered_extent_factors=" << ArrayToString(filtered_extent_factors) << ", "
            //                   "extent_to_factor=" << extent_to_factor;
            // }
          }
          if (factor_to_assign == 0) {
            std::uniform_int_distribution<> dist(1, max_extent);
            factor_to_assign = dist(*rng);
          }
          CHECK(factor_to_assign >= 1);
          // if (split_steps_info[iter_id].is_spatial) {
          //   reg_usage *= factor_to_assign;
          // } else {
          //   shmem_usage *= factor_to_assign;
          // }
          // max_extents.push_back(max_extent);
          factors_to_assign.push_back(factor_to_assign);
        }
        if (factors_to_assign.empty()) {
          CHECK(do_mutation);
          return;
        }
        // // shuffle the factors
        // std::vector<size_t> factors_to_assign_bak = factors_to_assign;
        // std::shuffle(factors_to_assign.begin(), factors_to_assign.end(), *rng);
        // // make sure that the shuffle is valid
        // bool valid_shuffle = true;
        // std::vector<size_t>::iterator
        //     max_extents_it = max_extents.begin(),
        //     factors_to_assign_it = factors_to_assign.begin();
        // // LOG(INFO) << "iter_max_extents=" << ArrayToString(iter_max_extents)
        // //           << " vs. factors_to_assign="
        // //           << ArrayToString(factors_to_assign);
        // for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
        //   if (fcontinue_predicate(iter_id)) {
        //     continue;
        //   }
        //   size_t iter_max_extent = *max_extents_it;
        //   if (*factors_to_assign_it > iter_max_extent) {
        //     valid_shuffle = false;
        //   }
        //   ++max_extents_it;
        //   ++factors_to_assign_it;
        // }
        // if (!valid_shuffle) {
        //   if (enable_verbose_logging) {
        //     LOG(WARNING) << "Shuffling is not valid";
        //   }
        //   factors_to_assign = std::move(factors_to_assign_bak);
        // }

        // do the actual assignment
        std::vector<size_t>::iterator factors_to_assign_it = factors_to_assign.begin();
        for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
          if (fcontinue_predicate(iter_id)) {
            continue;
          }
          CHECK(factors_to_assign_it != factors_to_assign.end());
          ffactor_to_assign(iter_id) = *factors_to_assign_it;
          ++factors_to_assign_it;
        }
        CHECK(factors_to_assign_it == factors_to_assign.end());
      };

  sample_factors(
        [&](const size_t iter_id) -> bool {
          return (!split_steps_info[iter_id].is_spatial) || 
                 (iter_id != last_spatial_iter_id);
        },
        [&](const size_t iter_id) -> size_t {
          // max_vthread_extent =
          //     std::min(max_vthread_extent,
          //              hardware_params->max_local_memory_per_block / reg_usage);
          return hardware_params->max_vthread_extent;
        },
        [&](const size_t iter_id) -> int& {
          return split_factors[iter_id][0];
        },
        [&](const size_t iter_id) -> size_t {
          return floor_div(split_steps_info[iter_id].max_extent,
                           split_factors[iter_id][1]
                 );
        }
      );

  if (enable_verbose_logging) {
    LOG(INFO) << "Finished sampling the vthread, current scheme=" << toString();
  }
  // ===========================================================================
  // factor[3] (innermost)
  // ===========================================================================
  sample_factors(
        [&](const size_t iter_id) -> bool {
          return (!split_steps_info[iter_id].is_spatial) || 
                 (iter_id == last_spatial_iter_id);
                 // always make sure that the final spatial axis has a stride of 1
        },
        [&](const size_t iter_id) -> size_t {
          // max_innermost_extent =
          //     std::min(max_innermost_extent,
          //              hardware_params->max_local_memory_per_block / reg_usage);
          return max_innermost_factor;
        },
        [&](const size_t iter_id) -> int& {
          return split_factors[iter_id][3];
        },
        [&](const size_t iter_id) -> size_t {
          return floor_div(
                   split_steps_info[iter_id].max_extent,
                   split_factors[iter_id][0] * split_factors[iter_id][1]
                 );
        }
      );
  if (enable_verbose_logging) {
    LOG(INFO) << "Finished sampling the innermost loop extent, current scheme="
              << toString();
  }

  // for (size_t iter_id = 0; iter_id < split_steps_info.size(); ++iter_id) {
  //   if (split_steps_info[iter_id].is_spatial) {
  //     shmem_usage += split_factors[iter_id][0] * split_factors[iter_id][1] *
  //                    split_factors[iter_id][2] * split_factors[iter_id][3];
  //   }
  // }
  // if (shmem_usage >
  //       static_cast<size_t>(
  //         hardware_params->max_shared_memory_per_block / sizeof(float)
  //       )
  //     ) {
  //   // LOG(FATAL) << "shmem_usage=" << shmem_usage << " is already greater than the allowable size "
  //   //              << hardware_params->max_shared_memory_per_block
  //   //              << ", current scheme=" << toString();
  //   throw std::out_of_range("shmem_usage goes out-of-range");
  // }
  // repeat similar procedure for reduction axes
  // ===========================================================================
  // rfactor[1] (innermost)
  // ===========================================================================
  sample_factors(
        [&](const size_t iter_id) -> bool {
          return split_steps_info[iter_id].is_spatial;
        },
        [&](const size_t iter_id) -> size_t {
          // max_innermost_extent =
          //     std::min(max_innermost_extent,
          //              hardware_params->max_shared_memory_per_block / sizeof(float) / shmem_usage);
          return max_innermost_factor;
        },
        [&](const size_t iter_id) -> int& {
          return split_factors[iter_id][1];
        },
        [&](const size_t iter_id) -> size_t {
          return split_steps_info[iter_id].max_extent;
        }
      );
  if (enable_verbose_logging) {
    LOG(INFO) << "Finished sampling the innermost loop extent (reduction axis), "
                 "current scheme=" << toString();
  }

  if (enable_verbose_logging) {
    LOG(INFO) << "Finished sampling the 2nd innermost loop extent (reduction axis), "
                 "current scheme=" << toString();
  }
  // If the sampled factorization scheme violates the hardware constraints, it
  // will just be discarded.
}


// FactorizationSchemeCheckRetType
// DietCodeSplitFactorizationMemo::IsLegit(const FactorizationScheme& scheme) {
//   int num_threads = 1;
//   size_t reg_usage = 1, acc_spatial = 0, acc_reduction = 1;

//   for (size_t i = 0; i < FactorizationScheme::split_steps_info.size(); ++i) {
//     const SplitStepInfo& split_step_info =
//         FactorizationScheme::split_steps_info[i];
//     const std::vector<int>& factors = scheme.split_factors[i];
//     if (split_step_info.is_spatial) {
//       CHECK(factors.size() == 4);
//       num_threads *= factors[1];
//       if (factors[0] > hardware_params_->max_vthread_extent) {
//         return kOOB;
//       }
//       if (factors[3] > max_innermost_factor_) {
//         return kOOB;
//       }
//       int64_t split_factors_prod =
//           factors[0] * factors[1] * factors[2] * factors[3];
//       if (split_factors_prod > split_step_info.max_extent) {
//         return kOOB;
//       }
//       reg_usage *= split_factors_prod;
//       acc_spatial += split_factors_prod;
//     } else {
//       CHECK(factors.size() == 2);
//       if (factors[1] > max_innermost_factor_) {
//         return kOOB;
//       }
//       int64_t split_factors_prod = factors[0] * factors[1];
//       if (split_factors_prod > split_step_info.max_extent) {
//         return kOOB;
//       }
//       acc_reduction *= split_factors_prod;
//     }
//   }
//   // check the shared memory capacity (with the assumption that all data types
//   // are 32-bit float)
//   size_t shmem_usage = acc_spatial * acc_reduction * sizeof(float);
//   if (shmem_usage >
//       static_cast<size_t>(hardware_params_->max_shared_memory_per_block)) {
//     return kOOB;
//   }
//   // check the number of threads per block
//   if (num_threads % hardware_params_->warp_size != 0) {
//     return kInvalid;
//   }
//   // LOG(INFO) << "scheme=" << scheme.toString();
//   // LOG(INFO) << "num_threads=" << num_threads << ", warp_size="
//   //           << hardware_params_->warp_size;
//   if (num_threads > hardware_params_->max_threads_per_block) {
//     return kOOB;
//   }
//   // check the number of registers used
//   if (reg_usage >
//       static_cast<size_t>(hardware_params_->max_local_memory_per_block)) {
//     return kOOB;
//   }
//   return kValid;
// }


// void DietCodeSplitFactorizationMemo::BfsEnumerate() {
//   while (!workstack_.empty()) {
//     FactorizationScheme& workitem = workstack_.front();
//     FactorizationSchemeCheckRetType ret = IsLegit(workitem);

//     if (ret == kValid) {
//       cache_.push_back(std::move(workitem));

//       if (cache_.size() % 10000 == 0) {
//         LOG(INFO) << "Number of valid schedules found so far: " << cache_.size();
//         LOG(INFO) << "Latest examined scheme=" << cache_.back().toString();
//       }
//     }
//     if (ret != kOOB) {
//       // consider expanding the factorization scheme
//       std::vector<FactorizationScheme> adj_schemes = workitem.neighbors();
//       for (FactorizationScheme& scheme : adj_schemes) {
//         if (examined_schemes_.find(scheme.toString()) ==
//             examined_schemes_.end()) {
//           examined_schemes_.insert(scheme.toString());
//           workstack_.push(std::move(scheme));

//           if (examined_schemes_.size() % 10000 == 0) {
//             LOG(INFO) << "Number of schemes examined so far: " << examined_schemes_.size();
//             LOG(INFO) << "Current workitem=" << workitem.toString();
//           }
//         }
//       }
//     }
//     workstack_.pop();
//   }
// }


// const std::vector<FactorizationScheme>&
// DietCodeSplitFactorizationMemo::GetAllFactorizationSchemes(
//     const std::vector<SplitStepInfo>& split_steps_info,
//     const bool simplify_sketch) {
//   if (!is_sample_init_population_1st_iter) {
//     CHECK(FactorizationScheme::split_steps_info == split_steps_info);
//     CHECK(FactorizationScheme::simplify_sketch == simplify_sketch);
//     // return the cached factorization scheme
//     return cache_;
//   }
//   workstack_.emplace(split_steps_info, simplify_sketch, true);
//   examined_schemes_.insert(workstack_.front().toString());
//   BfsEnumerate();
//   LOG(INFO) << "Total number of possible factorization schemes w/ the "
//                "dynamic workload: " << cache_.size();
//   return cache_;
// }

constexpr float C_FREEZE_NUM_THREADS = 0.8;
constexpr float C_SAMPLE_PERFECT_TILES_PROB = 1.;
static std::uniform_real_distribution<> uniform_real_disb(0.f, 1.f);

FactorizationScheme
DietCodeSplitFactorizationMemo::SampleFactorizationScheme(
    const std::vector<SplitStepInfo>& split_steps_info, std::mt19937* const rng
    ) {
  FactorizationScheme scheme// (split_steps_info, simplify_sketch,
                            //  is_sample_init_population_1st_iter
                            //  )
                             ;

  for (size_t i = 0; i < split_steps_info.size(); ++i) {
    scheme.split_factors.push_back(
        std::vector<int>(GetSplitLengths(split_steps_info[i].is_spatial), 1));
  }
  scheme.RandomSample(split_steps_info,
                      hardware_params_,
                      max_innermost_factor_,
                      rng,
                      false,  // do_mutation
                      true);
  
  if (enable_verbose_logging) {
    LOG(INFO) << "Randomly sampled factorization scheme=" << scheme.toString();
  }
  return scheme;
}


FactorizationScheme
DietCodeSplitFactorizationMemo::MutateFactorizationScheme(
    const std::vector<SplitStepInfo>& split_steps_info,
    std::mt19937* const rng,
    const std::vector<std::vector<int>>& curr_split_factors
    ) {
  // if (sample_perfect_tile_size) {
  //   LOG(INFO) << "Sampling perfect tile size";
  // } else {
  //   LOG(INFO) << "Sampling non-perfect tile size";
  // }

  FactorizationScheme scheme// (split_steps_info, simplify_sketch, true)
                            ;
  scheme.split_factors = curr_split_factors;
  scheme.RandomSample(split_steps_info,
                      hardware_params_,
                      max_innermost_factor_,
                      rng,
                      true,  // do_mutation
                      true
                      );

  if (enable_verbose_logging) {
    LOG(INFO) << "Randomly mutated factorization scheme=" << scheme.toString();
  }
  return scheme;
}


/********** Utils interface API for ffi **********/

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsGetConsumers")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id) {
      const std::set<int>& consumers = GetConsumers(task, state, stage_id);
      tvm::Map<IntImm, IntImm> ret;
      for (const auto& i : consumers) {
        ret.Set(Integer(i), Integer(i));
      }
      return ret;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsElementwiseMatch")
    .set_body_typed([](const SearchTask& task, const State& state, int stage_id,
                       int target_stage_id) {
      return ElementwiseMatch(task, state, stage_id, target_stage_id);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsIsTiled")
    .set_body_typed([](const Stage& stage) { return IsTiled(stage); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheReadStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheReadStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCacheWriteStage")
    .set_body_typed([](const State& s, int stage_id) { return HasCacheWriteStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasRfactorStage")
    .set_body_typed([](const State& s, int stage_id) { return HasRfactorStage(s, stage_id); });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction")
    .set_body_typed([](const State& s, int stage_id) {
      return HasCrossThreadReduction(s, stage_id);
    });

}  // namespace auto_scheduler
}  // namespace tvm
