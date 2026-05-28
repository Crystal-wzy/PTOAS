// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTO.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

SmallVector<unsigned> getSequentialEventIds(unsigned count) {
  SmallVector<unsigned> eventIds;
  eventIds.reserve(count);
  for (unsigned eventId = 0; eventId < count; ++eventId)
    eventIds.push_back(eventId);
  return eventIds;
}

void addPhase(SyncMacroModel &model, PipelineType pipe, ValueRange defValues,
              ValueRange useValues) {
  model.phases.push_back(SyncMacroPhase{
      static_cast<unsigned>(model.phases.size()), pipe,
      SmallVector<Value>(defValues.begin(), defValues.end()),
      SmallVector<Value>(useValues.begin(), useValues.end())});
}

void addHiddenEvent(SyncMacroModel &model, PipelineType srcPipe,
                    PipelineType dstPipe, ArrayRef<unsigned> eventIds) {
  model.hiddenEvents.push_back(
      SyncMacroHiddenEvent{srcPipe, dstPipe,
                           SmallVector<unsigned>(eventIds.begin(),
                                                 eventIds.end())});
}

void addBidirectionalHiddenEvent(SyncMacroModel &model, PipelineType firstPipe,
                                 PipelineType secondPipe,
                                 ArrayRef<unsigned> eventIds) {
  addHiddenEvent(model, firstPipe, secondPipe, eventIds);
  addHiddenEvent(model, secondPipe, firstPipe, eventIds);
}

std::optional<SyncMacroModel> getP2PCommSyncMacroModel(Operation *op) {
  Value dst;
  Value src;
  unsigned laneCount = 1;
  if (auto tput = dyn_cast<pto::TPutOp>(op)) {
    dst = tput.getDst();
    src = tput.getSrc();
    laneCount = tput.getPong() ? 2U : 1U;
  } else if (auto tget = dyn_cast<pto::TGetOp>(op)) {
    dst = tget.getDst();
    src = tget.getSrc();
    laneCount = tget.getPong() ? 2U : 1U;
  } else {
    return std::nullopt;
  }

  SyncMacroModel model;
  // P2P comm library calls first read the source GM through MTE2, then write
  // the destination GM through MTE3.
  addPhase(model, PipelineType::PIPE_MTE2, ValueRange{}, ValueRange{src});
  addPhase(model, PipelineType::PIPE_MTE3, ValueRange{dst}, ValueRange{});
  addBidirectionalHiddenEvent(model, PipelineType::PIPE_MTE2,
                              PipelineType::PIPE_MTE3,
                              getSequentialEventIds(laneCount));
  return model;
}

std::optional<SyncMacroModel> getCollectiveCommSyncMacroModel(Operation *op) {
  SyncMacroModel model;
  unsigned laneCount = 1;

  if (auto tgather = dyn_cast<pto::CommTGatherOp>(op)) {
    laneCount = tgather.getPong() ? 2U : 1U;
    // TGATHER_IMPL reads each group source through MTE2 and writes the gathered
    // result into dst through MTE3.
    addPhase(model, PipelineType::PIPE_MTE2, ValueRange{},
             tgather.getGroup());
    addPhase(model, PipelineType::PIPE_MTE3, ValueRange{tgather.getDst()},
             ValueRange{});
  } else if (auto tscatter = dyn_cast<pto::CommTScatterOp>(op)) {
    laneCount = tscatter.getPong() ? 2U : 1U;
    // TSCATTER_IMPL reads the source through MTE2 and writes every group
    // destination through MTE3.
    addPhase(model, PipelineType::PIPE_MTE2, ValueRange{},
             ValueRange{tscatter.getSrc()});
    addPhase(model, PipelineType::PIPE_MTE3, tscatter.getGroup(),
             ValueRange{});
  } else if (auto tbroadcast = dyn_cast<pto::TBroadcastOp>(op)) {
    laneCount = tbroadcast.getPong() ? 2U : 1U;
    // TBROADCAST_IMPL reads the source through MTE2 and writes every group
    // destination through MTE3.
    addPhase(model, PipelineType::PIPE_MTE2, ValueRange{},
             ValueRange{tbroadcast.getSrc()});
    addPhase(model, PipelineType::PIPE_MTE3, tbroadcast.getGroup(),
             ValueRange{});
  } else if (auto treduce = dyn_cast<pto::TReduceOp>(op)) {
    laneCount = treduce.getRecvPong() ? 3U : 2U;
    // TREDUCE_IMPL reads group sources through MTE2, reduces into acc on the
    // vector pipe, and stores the final result into dst through MTE3.
    addPhase(model, PipelineType::PIPE_MTE2, ValueRange{}, treduce.getGroup());
    addPhase(model, PipelineType::PIPE_V, ValueRange{treduce.getAcc()},
             ValueRange{treduce.getAcc(), treduce.getRecvPing()});
    addPhase(model, PipelineType::PIPE_MTE3, ValueRange{treduce.getDst()},
             ValueRange{});
  } else {
    return std::nullopt;
  }

  SmallVector<unsigned> eventIds = getSequentialEventIds(laneCount);
  addBidirectionalHiddenEvent(model, PipelineType::PIPE_MTE2,
                              PipelineType::PIPE_MTE3, eventIds);
  if (isa<pto::TReduceOp>(op)) {
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_V,
                   eventIds);
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_MTE2,
                   eventIds);
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_MTE3,
                   eventIds);
  }

  return model;
}

} // namespace

std::optional<SyncMacroModel> mlir::pto::getSyncMacroModel(Operation *op) {
  if (auto model = getP2PCommSyncMacroModel(op))
    return model;
  if (auto model = getCollectiveCommSyncMacroModel(op))
    return model;
  return std::nullopt;
}
