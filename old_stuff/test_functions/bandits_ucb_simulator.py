#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import math
from copy import copy
from typing import Callable, List

import numpy as np

DEFAULT_VARIANCES = [0.1, 0.4, 1.0, 2.0, 3.0]
DEFAULT_SHIFTS = [0, 20, 40, 60, 80]


def bandits_ucb_simulator(
    cs: List[float],
    n_reps: int,
    n_steps: int = 200,
    n_arms: int = 5,
    variances: int = DEFAULT_VARIANCES,
    shifts: int = DEFAULT_SHIFTS,
    threshold: float = -2.0,
    status_quo_cost: float = 0.1,
    cost_scale: float = 1.0,
    non_stat_f: Callable = None,
) -> [float, float]:
    r"""Simulates the UCB policy in a simple bandit problem.

    This function simulates the UCB policy in a bandit problem under independent
    normal rewards with means varying periodically.

    Args:
        cs: List of floats containing the confidence levels to be used by the
            UCB policy in different time intervals.
        n_reps: Number of replications.
        threshold: Real value indicating a the minimum acceptable expected
            reward. A cost (threshold - means[selected_arm])^+ is incurred when
            sselected arm is pulled.
        status_quo_cost: Positive real value indicating the cost over which one
            whishes to improve.

    Returns:
        The average cumulative regret and the average incurred cost minus the
        status quo.
    """

    if n_arms != len(variances) or n_arms != len(shifts):
        raise ValueError(
            "Number of arms should coincide with the number of variances and shifts."
        )

    num_params = len(cs)
    total_cumulative_regrets = []
    total_cumulative_costs = []

    if non_stat_f is None:

        def non_stat_f(t, shift):
            return np.sin(2 * np.pi * t / n_steps * 2 - shift)

    ts = np.arange(n_steps)
    variances = [0.1, 0.4, 1.0, 2.0, 3.0]
    shifts = [0, 20, 40, 60, 80]

    seasonalities = np.zeros((n_arms, n_steps))
    for i in range(n_arms):
        seasonalities[i, :] = non_stat_f(ts, shifts[i])

    # Generate all rewards for each arm
    # (this helps making the outputs smoother as a function of the policy parameters)
    for seed in range(n_reps):
        random_state = np.random.RandomState(seed)
        means_rnd = random_state.normal(size=n_arms)
        means = np.expand_dims(means_rnd, 1) + seasonalities
        max_reward = np.max(means, axis=0)
        rewards = []
        for arm in range(n_arms):
            rewards.append([])
            for t in range(n_steps):
                rewards[arm].append(
                    random_state.normal(
                        loc=means[arm, t], scale=np.sqrt(variances[arm])
                    )
                )

        arms_selected = []
        numbers_of_selections = [0 for _ in range(n_arms)]
        cumulative_rewards = [0.0 for _ in range(n_arms)]
        total_cumulative_regret = 0.0
        total_cumulative_cost = 0.0

        # Sample all arms once
        for arm in range(n_arms):
            arms_selected.append(copy(arm))
            numbers_of_selections[arm] += 1
            reward = random_state.normal(
                loc=means[arm, 0], scale=np.sqrt(variances[arm])
            )
            cumulative_rewards[arm] += reward
            total_cumulative_regret += means[arm, 0] - max_reward
            total_cumulative_cost += cost_scale * max(threshold - means[arm, 0], 0.0)

        # Run UCB for n_steps
        for t_i in range(n_steps - 1):
            t = t_i + 1
            c = cs[
                math.floor(t / (n_steps / num_params))
            ]  # Pick c to be used in this time interval
            ucbs = []
            # Compute UCBs
            for arm in range(n_arms):
                average_reward = cumulative_rewards[arm] / numbers_of_selections[arm]
                cb_width = c * np.sqrt(2 * np.log(t + 1) / numbers_of_selections[arm])
                ucbs.append(average_reward + cb_width)
            # Sample arm with highest UCB
            selected_arm = np.argmax(ucbs)
            arms_selected.append(selected_arm)
            numbers_of_selections[selected_arm] += 1
            reward = rewards[selected_arm][t]
            cumulative_rewards[selected_arm] += reward
            total_cumulative_regret += means[selected_arm, t] - max_reward[t]
            # Cost is the positive part of the thresold minus the expected reward of the chosen arm
            total_cumulative_cost += cost_scale * max(
                threshold - means[selected_arm, t], 0.0
            )
        total_cumulative_regrets.append(copy(total_cumulative_regret))
        total_cumulative_costs.append(copy(total_cumulative_cost))

    return (
        np.mean(total_cumulative_regrets),
        np.mean(total_cumulative_costs) - status_quo_cost,
    )
