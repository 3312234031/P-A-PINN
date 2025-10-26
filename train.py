
from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.fleet.utils import hybrid_parallel_util as hpu
from paddle.framework import core

from ppsci.solver import printer
from ppsci.utils import misc
import csv
import os 

if TYPE_CHECKING:
    from solver import Solver

loss_history = []

def initialize_csv(epoch_id):
    csv_filename = "training_loss.csv"

    if epoch_id == 1:
        if os.path.isfile(csv_filename):
            os.remove(csv_filename)  
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "iteration", "total_loss"] + 
                            ["constraint_loss_" + key for key in ["inlet", "outlet1", "outlet2", "outlet3", "outlet4", "outlet5", "outlet6", "noslip", "interior", "igc_outlet_combined", "igc_integral"]])


def save_loss_to_csv(epoch_id, iter_id, total_loss, loss_dict, log_freq):
    total_loss_value = total_loss.item() if hasattr(total_loss, 'item') else total_loss  

    csv_filename = "training_loss.csv"

    if epoch_id == 1 and iter_id == 1:
        loss_entry = [epoch_id, iter_id, total_loss_value] + list(loss_dict.values())
        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(loss_entry)

    elif iter_id % 1000 == 0:
        loss_entry = [epoch_id, iter_id, total_loss_value] + list(loss_dict.values())
        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(loss_entry)
            
def train_epoch_func(solver: "Solver", epoch_id: int, log_freq: int):

    batch_tic = time.perf_counter()

    for iter_id in range(1, solver.iters_per_epoch + 1):
        if solver.nvtx_flag:  # only for nsight analysis
            core.nvprof_nvtx_push(
                f"Training iteration {solver.global_step + 1}"
            )  # Training iteration

        total_batch_size = 0
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()

        input_dicts = []
        label_dicts = []
        weight_dicts = []
        for _, _constraint in solver.constraint.items():
            # fetch data from data loader
            try:
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            except StopIteration:
                _constraint.data_iter = iter(_constraint.data_loader)
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            reader_cost += time.perf_counter() - reader_tic

            for v in input_dict.values():
                if hasattr(v, "stop_gradient"):
                    v.stop_gradient = False

            # gather each constraint's input, label, weight to a list
            input_dicts.append(input_dict)
            label_dicts.append(label_dict)
            weight_dicts.append(weight_dict)
            total_batch_size += next(iter(input_dict.values())).shape[0]
            reader_tic = time.perf_counter()

        loss_dict = misc.Prettydefaultdict(float)
        loss_dict["loss"] = 0.0
        # forward for every constraint, including model and equation expression
        with solver.no_sync_context_manager(solver.world_size > 1, solver.model):
            with solver.autocast_context_manager(solver.use_amp, solver.amp_level):
                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_push("Loss computation")

                losses_all, losses_constraint = solver.forward_helper.train_forward(
                    tuple(
                        _constraint.output_expr
                        for _constraint in solver.constraint.values()
                    ),
                    input_dicts,
                    solver.model,
                    solver.constraint,
                    label_dicts,
                    weight_dicts,
                )
                assert "loss" not in losses_all, (
                    "Key 'loss' is not allowed in loss_dict for it is an preserved key"
                    " representing total loss, please use other name instead."
                )

                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_pop()  # Loss computation

                # accumulate all losses
                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_push("Loss aggregator")

                total_loss = solver.loss_aggregator(losses_all, solver.global_step)
                if solver.update_freq > 1:
                    total_loss = total_loss / solver.update_freq

                loss_dict.update(losses_constraint)
                loss_dict["loss"] = float(total_loss)

                if solver.nvtx_flag:  # only for nsight analysis
                    core.nvprof_nvtx_pop()  # Loss aggregator

            save_loss_to_csv(epoch_id, iter_id, total_loss, loss_dict, log_freq)

            # backward
            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_push("Loss backward")

            if solver.use_amp:
                total_loss_scaled = solver.scaler.scale(total_loss)
                total_loss_scaled.backward()
            else:
                total_loss.backward()

            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_pop()  # Loss backward

        # update parameters
        if iter_id % solver.update_freq == 0 or iter_id == solver.iters_per_epoch:
            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_push("Optimizer update")

            if solver.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP + no_sync
                # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                hpu.fused_allreduce_gradients(list(solver.model.parameters()), None)
            if solver.use_amp:
                solver.scaler.minimize(solver.optimizer, total_loss_scaled)
            else:
                solver.optimizer.step()

            if solver.nvtx_flag:  # only for nsight analysis
                core.nvprof_nvtx_pop()  # Optimizer update

            solver.optimizer.clear_grad()

        # update learning rate by step
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        if solver.benchmark_flag:
            paddle.device.synchronize()
        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if (
            solver.global_step % log_freq == 0
            or solver.global_step == 1
            or solver.global_step == solver.max_steps
        ):
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()

        if solver.nvtx_flag:  # only for nsight analysis
            core.nvprof_nvtx_pop()  # Training iteration
            NVTX_STOP_ITER = 25
            if solver.global_step >= NVTX_STOP_ITER:
                print(
                    f"Only run {NVTX_STOP_ITER} steps when 'NVTX' is set in environment"
                    " for nsight analysis. Exit now ......\n"
                )
                core.nvprof_stop()
                sys.exit(0)


def train_LBFGS_epoch_func(solver: "Solver", epoch_id: int, log_freq: int):

    batch_tic = time.perf_counter()

    for iter_id in range(1, solver.iters_per_epoch + 1):
        loss_dict = misc.Prettydefaultdict(float)
        loss_dict["loss"] = 0.0
        total_batch_size = 0
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()

        input_dicts = []
        label_dicts = []
        weight_dicts = []
        for _, _constraint in solver.constraint.items():
            # fetch data from data loader
            try:
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            except StopIteration:
                _constraint.data_iter = iter(_constraint.data_loader)
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
            reader_cost += time.perf_counter() - reader_tic

            for v in input_dict.values():
                if hasattr(v, "stop_gradient"):
                    v.stop_gradient = False

            # gather each constraint's input, label, weight to a list
            input_dicts.append(input_dict)
            label_dicts.append(label_dict)
            weight_dicts.append(weight_dict)
            total_batch_size += next(iter(input_dict.values())).shape[0]
            reader_tic = time.perf_counter()

        def closure() -> paddle.Tensor:
            """Forward-backward closure function for LBFGS optimizer.

            Returns:
                paddle.Tensor: Computed loss scalar.
            """
            with solver.no_sync_context_manager(solver.world_size > 1, solver.model):
                with solver.autocast_context_manager(solver.use_amp, solver.amp_level):
                    # forward for every constraint, including model and equation expression
                    losses_all, losses_constraint = solver.forward_helper.train_forward(
                        tuple(
                            _constraint.output_expr
                            for _constraint in solver.constraint.values()
                        ),
                        input_dicts,
                        solver.model,
                        solver.constraint,
                        label_dicts,
                        weight_dicts,
                    )

                    # accumulate all losses
                    total_loss = solver.loss_aggregator(losses_all, solver.global_step)
                    loss_dict.update(losses_constraint)
                    loss_dict["loss"] = float(total_loss)

                # backward
                solver.optimizer.clear_grad()
                total_loss.backward()

            if solver.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP model
                # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                hpu.fused_allreduce_gradients(list(solver.model.parameters()), None)

            return total_loss

        # update parameters
        solver.optimizer.step(closure)

        # update learning rate by step
        if solver.lr_scheduler is not None and not solver.lr_scheduler.by_epoch:
            solver.lr_scheduler.step()

        if solver.benchmark_flag:
            paddle.device.synchronize()
        batch_cost += time.perf_counter() - batch_tic

        # update and log training information
        solver.global_step += 1
        solver.train_time_info["reader_cost"].update(reader_cost)
        solver.train_time_info["batch_cost"].update(batch_cost)
        printer.update_train_loss(solver, loss_dict, total_batch_size)
        if (
            solver.global_step % log_freq == 0
            or solver.global_step == 1
            or solver.global_step == solver.max_steps
        ):
            printer.log_train_info(solver, total_batch_size, epoch_id, iter_id)

        batch_tic = time.perf_counter()
