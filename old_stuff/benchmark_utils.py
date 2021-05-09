#!/usr/bin/env python3

import logging
from contextlib import ExitStack
from copy import copy
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import fblearner.flow.api as flow

# @dep=//pytorch/botorch:botorch
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qSimpleRegret,
)

# @dep=//pytorch/botorch_fb:botorch_fb_experimental
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_step_lookahead import (
    qMultiStepLookahead,
    warmstart_multistep,
)
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    ScalarizedObjective,
)
from botorch.acquisition.utils import expand_trace_observations
from botorch.generation.gen import get_best_candidates
from botorch.generation.sampling import MaxPosteriorSampling, SamplingStrategy
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.outcome import (
    OutcomeTransform,
    Standardize,
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.sampling.samplers import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.transforms import normalize, unnormalize
from botorch_fb.acquisition.budget_constrained_multi_step_kg import (
    BudgetConstrainedMultiStepLookaheadKG,
)
from botorch_fb.acquisition.multi_fidelity_utils import fidelity_bound_adjustment
from botorch_fb.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch_fb.acquisition.mves import (
    qLowerBoundMaxValueEntropy,
    qLowerBoundMultiFidelityMaxValueEntropy,
)
from botorch_fb.experimental.acquisition.budget_constrained_multi_step_ei import (
    BudgetConstrainedMultiStepLookaheadEI,
)
from botorch_fb.experimental.acquisition.cost_aware_multi_step_lagrangian_ei import (
    CostAwareMultiStepLookaheadLagrangianEI,
)
from botorch_fb.experimental.acquisition.cost_aware_multi_step_lagrangian_kg import (
    qCostAwareMultiStepLookaheadLagrangianKG,
)
from botorch_fb.experimental.acquisition.cost_aware_multi_step_ratio_ei import (
    CostAwareMultiStepLookaheadRatioEI,
)
from botorch_fb.experimental.acquisition.cost_aware_multi_step_ratio_kg import (
    qCostAwareMultiStepLookaheadRatioKG,
)
from botorch_fb.experimental.acquisition.cost_aware_ts import CostAwareThompsonSampling
from botorch_fb.experimental.acquisition.cost_aware_ucb import (
    CostAwareUpperConfidenceBound,
)
from botorch_fb.experimental.acquisition.lagrangian_multi_step_ei import (
    LagrangianMultiStepLookaheadExpectedImprovement,
)
from botorch_fb.experimental.acquisition.multi_step_ei import MultiStepLookaheadEI
from botorch_fb.experimental.acquisition.naive_cost_aware import (
    NaiveCostAwareAcquisitionFunction,
)
from botorch_fb.experimental.stopping.probability_optimal import (
    NoisyProbabilityOptimal,
    ProbabilityOptimal,
)
from botorch_fb.models.fidelity.fidelity_utils import (
    cost_with_fidelity,
    warmstart_initialization,
)
from gpytorch import settings as gpt_settings
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.stats.mstats import winsorize as scipy_winsorize
from torch import Tensor

from .problems import ProblemSpec


TAlgoParams = Dict[str, Optional[Union[int, str, List[int]]]]

# multi-fidelity acquisition functions
MF_ALGOS_LIST = ["NCKGF", "qLB_MF_MVE", "qMF_MVE", "MF_MS_R_KG", "BC_MS_KG"]
# full-fidelity (fixed-feature) acquisition functions
FF_ALGOS_LIST = ["FF_EI", "FF_NEI"]
# cost-aware acquisition functions
CA_ALGOS_LIST = [
    "NCA_EI",
    "NCA_qEI",
    "CA_TS",
    "CA_UCB",
    "CA_MS_R_EI",
    "CA_MS_L_EI",
    "CA_MS_R_KG",
    "CA_MS_L_KG",
    "MF_MS_R_KG",
    "MF_MS_L_KG",
    "BC_MS_EI",
    "L_MS_EI",
    "BC_MS_KG",
    "CA_EI_cool",
]


class BOState(NamedTuple):
    func_evals: List[float]
    recs: List[List[float]]
    train_x: List[List[float]]
    train_obj: List[List[float]]


class OneSampleRun(NamedTuple):
    algos: List[str]
    func_evals: List[List[float]]
    recs: List[List[List[float]]]
    train_xs: List[List[List[float]]]
    train_objs: List[List[List[float]]]
    train_cons: List[List[List[float]]]
    train_costs: List[List[float]]
    wall_times: List[List[float]]
    acq_value_vec: List[List[float]]
    cost_evals: List[List[float]]
    failure_mode: str
    hvs: List[List[float]]
    stopping_information: List[List[Dict[str, Union[bool, float, str]]]]
    budgets: Optional[List[List[float]]] = None
    fantasy_costs: Optional[List] = None


def winsorize(y: Tensor, winsorization_level: float, maximize: bool = True) -> Tensor:
    if maximize:
        winsorize_limits = (winsorization_level, None)
    else:
        winsorize_limits = (None, winsorization_level)
    return torch.from_numpy(scipy_winsorize(y.cpu().numpy(), winsorize_limits)).to(y)


def initialize_model(
    train_x: Tensor,
    train_obj: Tensor,
    problem_spec: ProblemSpec,
    log_transform_obj: bool = False,
    train_con: Optional[Tensor] = None,
    state_dict: Optional[Dict] = None,
    model_cls: Optional[Type[GPyTorchModel]] = None,
    model_options: Optional[Dict[str, bool]] = None,
) -> Tuple[ExactMarginalLogLikelihood, Model, Optional[OutcomeTransform]]:
    model_options = model_options or {}
    if log_transform_obj:
        train_obj = torch.log(train_obj)
    if problem_spec.winsorization_level > 0:
        train_obj = winsorize(
            train_obj, winsorization_level=problem_spec.winsorization_level
        )
    if train_obj.ndim == 1:
        train_obj = train_obj.unsqueeze(-1)
    if train_con is not None:
        if train_con.ndim == 1:
            train_con = train_con.unsqueeze(-1)
        train_obj = torch.cat([train_obj, train_con], dim=-1)

    # standardize
    outcome_tf = Standardize(m=train_obj.shape[1])
    # apply tfs and set mean and std in Standardize
    train_obj, _ = outcome_tf(train_obj)
    outcome_tf.eval()
    model_args = {
        "train_X": normalize(train_x, bounds=problem_spec.bounds),
        "train_Y": train_obj,
    }
    if model_cls is None:
        if problem_spec.noise_se != 0.0:
            model_cls = SingleTaskGP
        else:
            model_cls = FixedNoiseGP
            model_args["train_Yvar"] = torch.full_like(model_args["train_Y"], 1e-5)
    if model_cls == SingleTaskMultiFidelityGP:
        fidelity_args = {
            a: model_options.get(a) for a in ("data_fidelity", "iteration_fidelity")
        }
        model_args.update(fidelity_args)
        model_args.update(
            {"linear_truncated": model_options.get("linear_truncated", False)}
        )
    model = model_cls(**model_args)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model, outcome_tf


def generate_random_designs(
    n: int,
    problem_spec: ProblemSpec,
    sobol: bool = True,
    sobol_seed: int = 0,
    cuda: bool = False,
    double: bool = True,
    fidelity_bound: Optional[List] = None,
) -> Tensor:
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.double if double else torch.float

    if sobol:
        soboleng = torch.quasirandom.SobolEngine(
            dimension=problem_spec.dimension, scramble=True, seed=sobol_seed
        )
        train_x = soboleng.draw(n).to(device=device, dtype=dtype)
    else:
        train_x = torch.rand(n, problem_spec.dimension, device=device, dtype=dtype)
    bnds = problem_spec.bounds
    if fidelity_bound is not None:
        bnds = fidelity_bound_adjustment(
            original_bounds=bnds,
            fidelity_dims=problem_spec.fidelity_dims,
            fidelity_bound=fidelity_bound,
        )
    train_x = (bnds[1] - bnds[0]) * train_x + bnds[0]
    return train_x


def generate_initial_data(
    n: int,
    problem_spec: ProblemSpec,
    sobol: bool = True,
    sobol_seed: int = 0,
    cuda: bool = False,
    double: bool = True,
    num_trace_observations: int = 0,
    fidelity_bound: Optional[List] = None,
    custom_fidelities: Optional[Dict] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """custom_fidelities = {f_1: 10, f_2: 20} means 10 points of the
    initial data use f_1 fidelity, and 20 points use f_2 fidelity, and
    the remainder are random fidelities. If custom_fidelities is None,
    we treat the fidelity parameter as a normal design parameter."""

    train_x = generate_random_designs(
        n=n,
        problem_spec=problem_spec,
        sobol=sobol,
        sobol_seed=sobol_seed,
        cuda=cuda,
        double=double,
        fidelity_bound=fidelity_bound,
    )
    if custom_fidelities is not None:
        # re-adjust the fidelity dimension
        inputs = torch.tensor(list(custom_fidelities.keys()))
        repeats = torch.tensor(list(custom_fidelities.values()))
        repeat_fid = torch.repeat_interleave(inputs, repeats).to(train_x)
        # shuffle them around
        for fdim in problem_spec.fidelity_dims:
            train_x[:, fdim] = repeat_fid[torch.randperm(n)]

    train_x = expand_trace_observations(
        X=train_x,
        fidelity_dims=problem_spec.fidelity_dims,
        num_trace_obs=num_trace_observations,
    )
    train_obj, train_con, train_cost = make_new_observation(train_x, problem_spec)
    logging.info(f"initial train_x = {train_x}")
    logging.info(f"initial train_obj = {train_obj}")
    logging.info(f"initial train_con = {train_con}")
    logging.info(f"initial train_cost = {train_cost}")
    return train_x, train_obj, train_con, train_cost


def get_acquisition_function(  # noqa C901
    model: Model,
    algo: str,
    algo_params: TAlgoParams,
    train_x: Tensor,
    train_obj: Tensor,
    bounds: Tensor,
    winsorization_level: float,
    mc_obj: Optional[GenericMCObjective] = None,
    train_con: Optional[Tensor] = None,
    constrained_obj: Optional[ConstrainedMCObjective] = None,
    train_cost: Optional[Tensor] = None,
    cost_model: Optional[Model] = None,
    cost_aware_utility: Optional[CostAwareUtility] = None,
    is_log_cost: Optional[bool] = False,
    ref_point: Optional[List[float]] = None,
    partitioning: Optional[NondominatedPartitioning] = None,
) -> Optional[Union[AcquisitionFunction, SamplingStrategy]]:
    is_constrained = train_con is not None and constrained_obj is not None
    if algo in ("EI", "FF_EI"):
        use_qmc = algo_params.get("use_qmc")
        mc_samples = algo_params.get("mc_samples")
        sampler = (
            SobolQMCNormalSampler(num_samples=mc_samples)
            if use_qmc
            else IIDNormalSampler(num_samples=mc_samples)
        )
        if not is_constrained:
            if winsorization_level > 0:
                train_obj = winsorize(
                    train_obj, winsorization_level=winsorization_level
                )
            if mc_obj is not None:
                best_f = mc_obj(train_obj).max()
            else:
                best_f = train_obj.max()

            acq_func = qExpectedImprovement(
                model=model, best_f=best_f, sampler=sampler, objective=mc_obj
            )
        else:
            # since we are using constrained_obj, which has unstandardized y's,
            # best_f should also be unstandardized.
            train_obj_feas = train_obj[(train_con <= 0).all(dim=-1)]
            if len(train_obj_feas) > 0:
                best_f = constrained_obj.objective(train_obj_feas).max().item()
            else:
                # no feasible observations, just take the minimum observation so far
                # TODO: make sure that objectives are in fact non-negative for single
                # objective optimization (ParEGO scales objectives to [0,1]^m).
                # T68584609
                best_f = constrained_obj.objective(train_obj).min().item()
            acq_func = qExpectedImprovement(
                model=model, best_f=best_f, sampler=sampler, objective=constrained_obj
            )
    elif algo == "analytic_EI":
        if is_constrained:
            raise Exception("Analytic EI does not support constraints.")
        if winsorization_level > 0:
            train_obj = winsorize(train_obj, winsorization_level=winsorization_level)
        best_f = train_obj.max().item()
        acq_func = ExpectedImprovement(model=model, best_f=best_f)
    elif algo == "NCKG" or algo == "NCKGF":
        use_qmc = algo_params.get("use_qmc")
        n_fantasies = algo_params.get("n_fantasies")
        sampler_cls = SobolQMCNormalSampler if use_qmc else IIDNormalSampler
        fantasy_sampler = sampler_cls(num_samples=n_fantasies)
        if algo == "NCKG":
            inner_mc_samples = algo_params.get("inner_mc_samples")
            sampler = sampler_cls(num_samples=inner_mc_samples)
            acq_func = qKnowledgeGradient(
                model=model,
                num_fantasies=n_fantasies,
                sampler=fantasy_sampler,
                objective=constrained_obj if is_constrained else mc_obj,
                inner_sampler=sampler,
            )
        else:
            current_value = algo_params.get("current_value")
            acq_func = qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=n_fantasies,
                sampler=fantasy_sampler,
                objective=mc_obj,
                current_value=current_value,
                cost_aware_utility=cost_aware_utility,
                project=algo_params["project"],
                expand=algo_params["expand"],
            )

    elif algo == "MSKG":
        raise NotImplementedError  # TODO: Update to new qMsLa api
        # batch_sizes = [algo_params.get("batch_size")] + algo_params.get(
        #     "lookahead_batch_sizes"
        # )
        # num_fantasies = [algo_params.get("n_fantasies")] + algo_params.get(
        #     "lookahead_n_fantasies"
        # )
        # is_analytic = batch_sizes[-1] == 1
        # acq_func = qMultiStepLookahead(
        #     model=model,
        #     batch_sizes=batch_sizes,
        #     num_fantasies=num_fantasies,
        #     objective=constrained_obj if is_constrained else mc_obj,
        #     value_function=PosteriorMean if is_analytic else qSimpleRegret,
        #     inner_sampler=(
        #         None
        #         if is_analytic
        #         else SobolQMCNormalSampler(
        #             num_samples=algo_params.get("inner_mc_samples"),
        #             resample=False,
        #             collapse_batch_dims=True,
        #         )
        #     ),
        # )
    elif algo in ("NEI", "FF_NEI"):
        use_qmc = algo_params.get("use_qmc")
        mc_samples = algo_params.get("mc_samples")
        sampler = (
            SobolQMCNormalSampler(num_samples=mc_samples)
            if use_qmc
            else IIDNormalSampler(num_samples=mc_samples)
        )
        acq_func = qNoisyExpectedImprovement(
            model=model,
            X_baseline=normalize(train_x, bounds=bounds),
            sampler=sampler,
            objective=constrained_obj if is_constrained else mc_obj,
            prune_baseline=True,
        )
    elif algo == "qLB_MVE":
        mc_samples = algo_params.get("mc_samples", 10)
        num_samples = algo_params.get("num_mv_samples", 20)
        candidate_size = algo_params.get("candidate_size", 100)
        use_gumbel = algo_params.get("use_gumbel", True)
        acq_func = qLowerBoundMaxValueEntropy(
            model=model,
            bounds=bounds,
            num_samples=num_samples,
            mc_samples=mc_samples,
            candidate_size=candidate_size,
            use_gumbel=use_gumbel,
        )
    elif algo == "qLB_MF_MVE":
        mc_samples = algo_params.get("mc_samples", 10)
        num_samples = algo_params.get("num_mv_samples", 20)
        candidate_size = algo_params.get("candidate_size", 100)
        use_gumbel = algo_params.get("use_gumbel", True)
        ncols_fidelity = len(algo_params.get("fidelity_dims"))
        acq_func = qLowerBoundMultiFidelityMaxValueEntropy(
            model=model,
            bounds=bounds,
            cost=cost_with_fidelity,
            num_samples=num_samples,
            mc_samples=mc_samples,
            candidate_size=candidate_size,
            use_gumbel=use_gumbel,
            ncols_fidelity=ncols_fidelity,
        )
    elif algo == "qMVE":
        mc_samples = algo_params.get("mc_samples", 10)
        num_samples = algo_params.get("num_mv_samples", 20)
        n_fantasies = algo_params.get("n_fantasies", 10)
        cand_set_size = algo_params.get("candidate_size", 100)
        candidate_set = torch.rand(cand_set_size, bounds.size(1))
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        use_gumbel = algo_params.get("use_gumbel", True)
        acq_func = qMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,
            num_mv_samples=num_samples,
            num_y_samples=mc_samples,
            num_fantasies=n_fantasies,
            use_gumbel=use_gumbel,
        )
    elif algo == "qMF_MVE":
        mc_samples = algo_params.get("mc_samples", 10)
        num_samples = algo_params.get("num_mv_samples", 20)
        n_fantasies = algo_params.get("n_fantasies", 10)
        cand_set_size = algo_params.get("candidate_size", 100)
        candidate_set = torch.rand(cand_set_size, bounds.size(1))
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        use_gumbel = algo_params.get("use_gumbel", True)
        acq_func = qMultiFidelityMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,
            num_mv_samples=num_samples,
            num_y_samples=mc_samples,
            num_fantasies=n_fantasies,
            use_gumbel=use_gumbel,
            cost_aware_utility=algo_params.get("cost_with_fidelity", None),
            project=algo_params.get("project", lambda X: X),
            expand=algo_params.get("expand", lambda X: X),
        )
    elif algo == "qEHVI":
        use_qmc = algo_params.get("use_qmc")
        mc_samples = algo_params.get("mc_samples")
        sampler = (
            SobolQMCNormalSampler(num_samples=mc_samples)
            if use_qmc
            else IIDNormalSampler(num_samples=mc_samples)
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=mc_obj,
            constraints=None if not is_constrained else constrained_obj.constraints,
        )
    elif algo == "qNEHVI":
        use_qmc = algo_params.get("use_qmc")
        mc_samples = algo_params.get("mc_samples")
        sampler = (
            SobolQMCNormalSampler(num_samples=mc_samples)
            if use_qmc
            else IIDNormalSampler(num_samples=mc_samples)
        )
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=normalize(train_x, bounds=bounds),
            sampler=sampler,
            objective=mc_obj,
            constraints=None if not is_constrained else constrained_obj.constraints,
            prune_baseline=True,
        )
    elif algo == "EHVI":
        acq_func = ExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            objective=mc_obj,
        )
    elif algo == "NCA_EI":
        if is_constrained:
            raise Exception("Analytic EI does not support constraints.")
        if winsorization_level > 0:
            train_obj = winsorize(train_obj, winsorization_level=winsorization_level)
        best_f = train_obj.max().item()
        raw_acq_func = ExpectedImprovement(model=model, best_f=best_f)
        use_mean = True
        min_cost = None if is_log_cost else torch.min(train_cost)

        acq_func = NaiveCostAwareAcquisitionFunction(
            raw_acqf=raw_acq_func,
            cost_model=cost_model,
            use_mean=use_mean,
            min_cost=min_cost,
        )
    elif algo == "NCA_qEI":
        use_qmc = algo_params.get("use_qmc")
        mc_samples = algo_params.get("mc_samples")
        sampler = (
            SobolQMCNormalSampler(num_samples=mc_samples)
            if use_qmc
            else IIDNormalSampler(num_samples=mc_samples)
        )
        if not is_constrained:
            if winsorization_level > 0:
                train_obj = winsorize(
                    train_obj, winsorization_level=winsorization_level
                )
            if mc_obj is not None:
                best_f = mc_obj(train_obj).max().item()
            else:
                best_f = train_obj.max().item()

            raw_acq_func = qExpectedImprovement(
                model=model.models[0], best_f=best_f, sampler=sampler, objective=mc_obj
            )
        else:
            # since we are using constrained_obj, which has unstandardized y's,
            # best_f should also be unstandardized.
            train_obj_feas = train_obj[(train_con <= 0).all(dim=-1)]
            if len(train_obj_feas) > 0:
                best_f = constrained_obj.objective(train_obj_feas).max().item()
            else:
                # no feasible observations, just take the minimum observation so far
                # TODO: make sure that objectives are in fact non-negative for single
                # objective optimization (ParEGO scales objectives to [0,1]^m).
                # T68584609
                best_f = constrained_obj.objective(train_obj).min().item()
            raw_acq_func = qExpectedImprovement(
                model=model.models[0],
                best_f=best_f,
                sampler=sampler,
                objective=constrained_obj,
            )
        use_mean = True
        min_cost = None if is_log_cost else torch.min(train_cost)

        acq_func = NaiveCostAwareAcquisitionFunction(
            raw_acqf=raw_acq_func,
            cost_model=model.models[1],
            use_mean=use_mean,
            min_cost=min_cost,
        )
    elif algo == "CA_TS":
        reference_val = algo_params.get("current_value")
        min_cost = None if is_log_cost else torch.min(train_cost)
        acq_func = CostAwareThompsonSampling(
            model=model, reference_val=reference_val, min_cost=min_cost
        )
    elif algo == "CA_UCB":
        beta_obj = algo_params.get("beta_obj")
        beta_cost = algo_params.get("beta_cost")
        reference_val = algo_params.get("current_value")
        min_cost = None if is_log_cost else torch.min(train_cost)

        acq_func = CostAwareUpperConfidenceBound(
            model=model,
            beta_obj=beta_obj,
            beta_cost=beta_cost,
            is_log_cost=is_log_cost,
            reference_val=reference_val,
            min_cost=min_cost,
        )
    elif algo == "CA_MS_R_EI":
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        min_cost = None if is_log_cost else torch.min(train_cost)

        acq_func = CostAwareMultiStepLookaheadRatioEI(
            model=model, min_cost=min_cost, num_fantasies=lookahead_num_fantasies
        )
    elif algo == "CA_MS_L_EI":
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        multiplier = copy(algo_params.get("multiplier"))
        budget = algo_params.get("budget")

        acq_func = CostAwareMultiStepLookaheadLagrangianEI(
            model=model,
            multiplier=multiplier,
            budget=budget,
            num_fantasies=lookahead_num_fantasies,
        )
    elif algo == "CA_MS_R_KG":
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        current_max_value = algo_params.get("current_value")
        min_cost = None if is_log_cost else torch.min(train_cost)

        acq_func = qCostAwareMultiStepLookaheadRatioKG(
            model=model,
            batch_sizes=lookahead_batch_sizes,
            current_max_value=current_max_value,
            min_cost=min_cost,
            num_fantasies=lookahead_num_fantasies,
        )
    elif algo == "CA_MS_L_KG":
        multiplier = copy(algo_params.get("multiplier"))
        budget = algo_params.get("budget")
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")

        acq_func = qCostAwareMultiStepLookaheadLagrangianKG(
            model=model,
            multiplier=multiplier,
            budget=budget,
            batch_sizes=lookahead_batch_sizes,
            num_fantasies=lookahead_num_fantasies,
        )
    elif algo == "MF_MS_R_KG":
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        current_max_value = algo_params.get("current_value")
        min_cost = None if is_log_cost else torch.min(train_cost)
        project = algo_params["project"]

        acq_func = qCostAwareMultiStepLookaheadRatioKG(
            model=model,
            batch_sizes=lookahead_batch_sizes,
            current_max_value=current_max_value,
            min_cost=min_cost,
            num_fantasies=lookahead_num_fantasies,
            project=project,
        )
    elif algo == "MF_MS_L_KG":
        multiplier = copy(algo_params.get("multiplier"))
        budget = algo_params.get("budget")
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        project = algo_params["project"]

        acq_func = qCostAwareMultiStepLookaheadLagrangianKG(
            model=model,
            multiplier=multiplier,
            budget=budget,
            batch_sizes=lookahead_batch_sizes,
            num_fantasies=lookahead_num_fantasies,
            project=project,
        )
    elif algo == "BC_MS_EI":
        budget = copy(algo_params.get("budget"))
        batch_size = algo_params.get("batch_size")
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        soft_plus_transform_budget = algo_params.get("soft_plus_transform_budget")
        if soft_plus_transform_budget:
            beta = 2.0 / train_cost.min().item()
        else:
            beta = None

        acq_func = BudgetConstrainedMultiStepLookaheadEI(
            model=model,
            budget=budget,
            batch_size=batch_size,
            lookahead_batch_sizes=lookahead_batch_sizes,
            num_fantasies=lookahead_num_fantasies,
            soft_plus_transform_budget=soft_plus_transform_budget,
            beta=beta,
        )
    elif algo == "L_MS_EI":
        multiplier = copy(algo_params.get("multiplier"))
        batch_size = algo_params.get("batch_size")
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")

        acq_func = LagrangianMultiStepLookaheadExpectedImprovement(
            model=model,
            multiplier=multiplier,
            batch_size=batch_size,
            lookahead_batch_sizes=lookahead_batch_sizes,
            num_fantasies=lookahead_num_fantasies,
        )
    elif algo == "TS":
        acq_func = MaxPosteriorSampling(
            model=model, objective=constrained_obj if is_constrained else mc_obj
        )
    elif algo == "BC_MS_KG":
        budget = copy(algo_params.get("budget"))
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")
        project = algo_params["project"]

        acq_func = BudgetConstrainedMultiStepLookaheadKG(
            model=model,
            budget=budget,
            num_fantasies=lookahead_num_fantasies,
            project=project,
        )
    elif algo == "MS_EI":
        batch_size = algo_params.get("batch_size")
        lookahead_batch_sizes = algo_params.get("lookahead_batch_sizes")
        lookahead_num_fantasies = algo_params.get("lookahead_n_fantasies")

        acq_func = MultiStepLookaheadEI(
            model=model,
            batch_size=batch_size,
            lookahead_batch_sizes=lookahead_batch_sizes,
            num_fantasies=lookahead_num_fantasies,
        )
    elif algo == "CA_EI_cool":
        if is_constrained:
            raise Exception("Analytic EI does not support constraints.")
        if winsorization_level > 0:
            train_obj = winsorize(train_obj, winsorization_level=winsorization_level)
        best_f = train_obj.max().item()
        raw_acq_func = ExpectedImprovement(model=model, best_f=best_f)
        use_mean = True
        cost_exponent = algo_params.get("cost_exponent")
        min_cost = None if is_log_cost else torch.min(train_cost)

        acq_func = NaiveCostAwareAcquisitionFunction(
            raw_acqf=raw_acq_func,
            cost_model=cost_model,
            use_mean=use_mean,
            cost_exponent=cost_exponent,
            min_cost=min_cost,
        )
    else:
        acq_func = None

    if algo in FF_ALGOS_LIST and (acq_func is not None):
        d = train_x.shape[-1]
        fidelity_dims = algo_params.get("fidelity_dims")
        acq_func = FixedFeatureAcquisitionFunction(
            acq_function=acq_func,
            d=d,
            columns=fidelity_dims,
            values=[1.0] * len(fidelity_dims),
        )

    return acq_func


def optimize_acqf_bm(  # noqa C901
    acq_func: Union[AcquisitionFunction, List[AcquisitionFunction]],
    problem_spec: ProblemSpec,
    batch_size: List[int],
    n_fantasies: Optional[List[int]] = None,
    raw_samples: int = 1000,
    num_restarts: int = 20,
    batch_limit: int = 5,
    nonnegative: bool = True,
    gpytorch_fast: bool = False,
    cholesky_max: int = 128,
    max_iter: int = 200,
    kg_options: Optional[Dict[str, Union[bool, Tensor, int]]] = None,
    return_best_only: bool = True,
    sequential: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Optimizes the acquisition function, and returns a new candidate."""
    is_kg = isinstance(acq_func, qKnowledgeGradient)
    is_ms = isinstance(acq_func, qMultiStepLookahead)
    dimension = problem_spec.dimension
    bounds = problem_spec.bounds
    fidelity_dims = problem_spec.fidelity_dims

    kg_options = kg_options or {}
    if isinstance(acq_func, FixedFeatureAcquisitionFunction):
        dimension -= len(fidelity_dims)
    standard_bounds = torch.tensor([[0.0] * dimension, [1.0] * dimension]).to(bounds)
    batch_initial_conditions = None
    q = batch_size[0]
    return_full_tree = False

    if is_kg:
        value_maximizer = kg_options.get("value_maximizer")
        partial_restarts = kg_options.get("partial_restarts")
        sampling_weights = kg_options.get("sampling_weights")
        # adjust the lower bound of fidelity parameters during optimization
        fidelity_bound = kg_options.get("fidelity_bound")
        if fidelity_bound is not None:
            standard_bounds = fidelity_bound_adjustment(
                original_bounds=standard_bounds,
                fidelity_dims=fidelity_dims,
                fidelity_bound=fidelity_bound,
            )
        if partial_restarts is not None:
            batch_initial_conditions = warmstart_initialization(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=q,  # augmenting q is handled automatically for vanilla KG
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                value_maximizer=value_maximizer,
                partial_restarts=partial_restarts,
                weights=sampling_weights,
            )
    if is_ms:
        q = acq_func.get_augmented_q_batch_size(q)
        return_full_tree = True

        if kg_options.get("full_optimizer") is not None:
            batch_initial_conditions = warmstart_multistep(
                acq_function=acq_func,
                bounds=standard_bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                full_optimizer=kg_options.get("full_optimizer"),
            )

    if isinstance(acq_func, qMultiFidelityKnowledgeGradient) or is_ms:
        return_best_only_adjusted = False  # set this to extract full soln
    else:
        return_best_only_adjusted = return_best_only

    fast_comp_settings = {
        "covar_root_decomposition": gpytorch_fast,
        "log_prob": gpytorch_fast,
        "solves": gpytorch_fast,
    }
    with ExitStack() as es:
        es.enter_context(gpt_settings.fast_pred_var())
        es.enter_context(gpt_settings.fast_computations(**fast_comp_settings))
        es.enter_context(gpt_settings.max_cholesky_size(cholesky_max))

        if isinstance(acq_func, AcquisitionFunction):
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={
                    "batch_limit": batch_limit,
                    "maxiter": max_iter,
                    "nonnegative": nonnegative,
                    "method": "L-BFGS-B",
                },
                batch_initial_conditions=batch_initial_conditions,
                return_best_only=return_best_only_adjusted,
                sequential=sequential or isinstance(acq_func, qMaxValueEntropy),
                return_full_tree=return_full_tree,
            )
        else:
            candidates, acq_values = optimize_acqf_list(
                acq_function_list=acq_func,
                bounds=standard_bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={
                    "batch_limit": batch_limit,
                    "maxiter": max_iter,
                    "nonnegative": nonnegative,
                    "method": "L-BFGS-B",
                },
            )

    if is_ms:
        # save all tree variables for multi-step initialization
        kg_options["full_optimizer"] = candidates.detach().clone()

    if (
        isinstance(acq_func, qMultiFidelityKnowledgeGradient) or is_ms
    ) and return_best_only:
        candidates = get_best_candidates(
            batch_candidates=candidates, batch_values=acq_values
        )

    if is_ms:
        kg_options["suggested_x_full_tree"] = candidates.detach().clone()

    new_x = (
        acq_func.extract_candidates(candidates.detach())
        if is_ms  # this happens in optimize_acqf for vanilla KG
        else candidates.detach()
    )
    if isinstance(acq_func, FixedFeatureAcquisitionFunction):
        new_x = acq_func._construct_X_full(new_x)
    return new_x, acq_values


def make_new_observation(
    new_x: Tensor, problem_spec: ProblemSpec
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

    if problem_spec.obj_cost_function is not None:
        obj, cost = problem_spec.obj_cost_function(new_x)
        con = None
        return obj, con, cost

    obj = problem_spec.objective(new_x)
    con = None
    cost = None
    num_objectives = 1 if obj.ndim == 1 else obj.shape[-1]
    noise_se = None
    if problem_spec.noise_se is not None:
        noise_se = torch.tensor(
            problem_spec.noise_se, dtype=obj.dtype, device=obj.device
        ).view(-1)
        obj += noise_se[:num_objectives] * torch.randn_like(obj)
    outcome_constraints = problem_spec.outcome_constraints
    if outcome_constraints is not None and len(outcome_constraints) > 0:
        # TODO: Use ConstrainedBaseTestProblem's evaluate_slack instead
        con = torch.stack([oc(new_x) for oc in outcome_constraints], dim=-1)
        # TODO: allow different noise levels for different outcomess
        if problem_spec.noise_se is not None:
            con += noise_se[-len(outcome_constraints) :] * torch.randn_like(con)
    if problem_spec.cost_function is not None:
        cost = problem_spec.cost_function(new_x)
        # TODO: this assumes that the cost is the last output
        if problem_spec.noise_se is not None:
            cost = torch.exp(torch.log(cost) + noise_se[-1] * torch.randn_like(cost))

    return obj, con, cost


def get_best_point(
    model: Model,
    algo: str,
    algo_params: TAlgoParams,
    general_params: Dict,
    problem_spec: ProblemSpec,
    train_x: Tensor,
    train_obj: Tensor,
    return_best_only: bool = True,
    train_con: Optional[Tensor] = None,
    objective: Optional[ScalarizedObjective] = None,
    constrained_obj: Optional[ConstrainedMCObjective] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    is_constrained = train_con is not None and constrained_obj is not None

    if algo_params.get("rec_strategy") == "out-of-sample":
        # make a recommendation by optimizing current model
        if not is_constrained:
            if algo in MF_ALGOS_LIST or algo in FF_ALGOS_LIST:
                acqf_post = PosteriorMean(model, objective=objective)
                best_point_acqf = FixedFeatureAcquisitionFunction(
                    acq_function=acqf_post,
                    d=problem_spec.dimension,
                    columns=problem_spec.fidelity_dims,
                    values=[1] * len(problem_spec.fidelity_dims),
                )
            else:
                best_point_acqf = PosteriorMean(model, objective=objective)
        else:
            qmc_sampler = SobolQMCNormalSampler(num_samples=1024)
            best_point_acqf = qSimpleRegret(
                model=model, sampler=qmc_sampler, objective=constrained_obj
            )
        new_rec_x, acq_value = optimize_acqf_bm(
            acq_func=best_point_acqf,
            problem_spec=problem_spec,
            batch_size=[1],
            raw_samples=general_params.get("best_point_opt_raw_samples"),
            num_restarts=general_params.get("best_point_opt_num_restarts"),
            batch_limit=general_params.get("best_point_opt_batch_limit"),
            nonnegative=False,
            gpytorch_fast=False,
            cholesky_max=general_params.get("cholesky_max"),
            max_iter=general_params.get("opt_iter"),
            return_best_only=return_best_only,
        )

        # the new_rec_x returned by optimize_acqf_bm is normalized to [0, 1]^d
        # squeeze q-batch dimension
        new_rec_x = unnormalize(new_rec_x, bounds=problem_spec.bounds).squeeze(dim=0)

        # clear gpytorch model caches to avoid issues with using the model
        # again later in a different acquisition function
        model.train().eval()
    elif algo_params.get("rec_strategy") == "in-sample":
        # make a recommendation based on previously observed points
        # TODO: for full-fidelity (FF) acquisition functions, only use FF
        # observed points. This issue is sidestepped if initial points are FF
        obj = train_obj
        if is_constrained:
            obj = obj * (train_con <= 0).all(dim=-1).to(obj)
        maxidx = obj.argmax()
        new_rec_x = train_x[maxidx]
        acq_value = train_obj[maxidx]
    else:
        raise Exception("Recommendation strategy not recognized.")
    return new_rec_x, acq_value


def gen_candidates(
    acq_func: Union[AcquisitionFunction, List[AcquisitionFunction], SamplingStrategy],
    algo: str,
    algo_params: TAlgoParams,
    general_params: Dict,
    problem_spec: ProblemSpec,
    cuda: bool,
    double: bool,
    kg_options: Optional[Dict[str, Union[bool, Tensor, int]]] = None,
    return_best_only: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    if algo == "RND":
        new_x = generate_random_designs(
            n=algo_params.get("batch_size", 1),
            problem_spec=problem_spec,
            sobol=False,  # generate uniformly random design
            cuda=cuda,
            double=double,
        )
        return new_x, torch.tensor([-1], dtype=new_x.dtype, device=new_x.device)
    elif algo == "TS":
        if isinstance(acq_func, SamplingStrategy):
            acq_func = [acq_func]
        discretization_size_per_dim = algo_params.get("discretization_size_per_dim")
        n_discrete_points = discretization_size_per_dim * problem_spec.dimension
        candidates = []
        for acqf in acq_func:
            discrete_X = torch.rand(
                n_discrete_points,
                problem_spec.dimension,
                dtype=torch.double if double else torch.float,
                device=torch.device("cuda" if cuda else "cpu"),
            )
            candidate = acqf(discrete_X, num_samples=1)
            candidates.append(candidate)
        candidates = torch.cat(candidates, dim=0)
        new_x = unnormalize(candidates, bounds=problem_spec.bounds)
        acq_value = None
    else:
        gpytorch_fast = general_params.get("gpytorch_fast")
        n_fantasies = [algo_params.get("n_fantasies")]
        batch_size = [algo_params.get("batch_size")]
        new_x, acq_value = optimize_acqf_bm(
            acq_func=acq_func,
            problem_spec=problem_spec,
            batch_size=batch_size,
            n_fantasies=n_fantasies,  # [None] if not NCKG of MSKG
            raw_samples=general_params.get("acq_opt_raw_samples"),
            num_restarts=general_params.get("acq_opt_num_restarts"),
            batch_limit=general_params.get("acq_opt_batch_limit"),
            nonnegative=algo not in ("NCKGF", "MSKG"),
            gpytorch_fast=gpytorch_fast,
            cholesky_max=1
            if algo in ("NCKG", "NCKGF", "MSKG", "CA_MS_R_KG", "MF_MS_R_KG")
            else general_params.get("cholesky_max"),  # for KG, always do fast
            kg_options=kg_options,
            return_best_only=return_best_only,
            sequential=general_params.get("sequential", False),
        )

        # the new_x returned by optimize_acqf is normalized to [0, 1]^d
        new_x = unnormalize(new_x, bounds=problem_spec.bounds)

        # clear gpytorch model caches to avoid issues with using the model
        # again later in a different acquisition function
        if isinstance(acq_func, AcquisitionFunction):
            acq_func.train().eval()
        else:
            [a.train().eval() for a in acq_func]

    return new_x, acq_value


def get_constrained_mc_objective(
    train_obj: Tensor,
    train_con: Tensor,
    scalarization: Optional[Callable[[Tensor], Tensor]] = None,
) -> ConstrainedMCObjective:
    n_obj = 1 if train_obj.ndim == 1 else train_obj.shape[-1]
    if n_obj == 1:
        if scalarization is not None:
            raise RuntimeError("Must have multiple objectives to use `scalarization`")

        # assume first outcome of the model is the objective, the rest constraints
        def objective(Z: Tensor) -> Tensor:
            return Z[..., 0]

    elif scalarization is None:
        # use ConstrainedMCObjective as a container for constraints only
        objective = None
    else:

        # assume first outcomes of the model are the objectives, the rest constraints
        def objective(Z: Tensor) -> Tensor:
            return scalarization(Z[..., :n_obj])

    def make_indexer(i: int) -> Callable[[Tensor], Tensor]:
        def indexer_i(Z: Tensor) -> Tensor:
            return Z[..., n_obj + i]

        return indexer_i

    constrained_obj = ConstrainedMCObjective(
        objective=objective,
        constraints=[make_indexer(i) for i in range(train_con.shape[-1])],
    )
    return constrained_obj


def get_learned_cost_mc_objective(train_cost: Tensor) -> GenericMCObjective:
    # in the case of learned cost (nonnegative), we model the log cost; there
    # is also a need to unstandardize / take exponential
    log_cost_std = torch.log(train_cost).std(dim=0).clamp_min(1e-6)
    log_cost_mean = torch.log(train_cost).mean(dim=0)
    mc_obj = GenericMCObjective(
        objective=lambda Z, X: torch.exp(
            unstandardize(Z[..., 0], log_cost_mean, log_cost_std)
        )
    )
    return mc_obj


def get_unstandardize_mc_objective(train_obj: Tensor) -> GenericMCObjective:
    obj_std = train_obj.std(dim=0).clamp_min(1e-6)
    obj_mean = train_obj.mean(dim=0)
    mc_obj = GenericMCObjective(
        objective=lambda Z, X: unstandardize(Z[..., 0], obj_mean, obj_std)
    )
    return mc_obj


def unstandardize(X: Tensor, data_mean: Tensor, data_std: Tensor) -> Tensor:
    return (X * data_std) + data_mean


def torch_to_list(xs: List[Tensor]) -> List[List[float]]:
    return [x.tolist() for x in xs]


def warmstart_init_multistep(
    acq_function: qMultiStepLookahead,
    value_maximizer: Tensor,
    Y_obs: Tensor,
    full_optimizer: Optional[Tensor],
    num_restarts: int,
    lmbda: float = 0.33,
) -> Tuple[int, Tensor]:
    r"""Warm-start initialization for multi-step look-ahead acquisition funcitons."""
    batch_sizes = acq_function.batch_sizes
    num_fantasies = acq_function.num_fantasies
    nfs = torch.tensor(num_fantasies)
    qtilde = sum(q * torch.prod(nfs[:i]).item() for i, q in enumerate(batch_sizes))

    # if this is the first step, we do a semi-cold start
    # if full_optimizer is None:
    vm_exp = value_maximizer.expand(num_restarts, qtilde, value_maximizer.shape[-1])
    # for exploration, sprinkle in some randomization (needs MUCH improvement)
    rndmask = torch.rand_like(vm_exp) < lmbda
    rndvals = torch.rand_like(vm_exp)
    batch_ics = (~rndmask).to(vm_exp) * vm_exp + rndmask.to(vm_exp) * rndvals
    return qtilde, batch_ics

    # TODO: Update with smarter intialization strategy utilizing the prev. tree
    # Xopts = split_ms_candidates(
    #     batch_sizes=batch_sizes, num_fantasies=num_fantasies, Xopt=full_optimizer
    # )
    # with torch.no_grad():
    #     fantasy_model = acq_function.model.fantasize(
    #         Xopts[0], sampler=acq_function.samplers[0],
    #     )
    #     # Nf x batch_shape x q0
    #     fantasy_samples = fantasy_model.train_targets[..., -batch_sizes[0]:]
    #
    # distances = torch.norm(fantasy_samples - Y_obs, dim=-1)
    # mindist = torch.argmin(distances, dim=0)
    # idxr = torch.arange(len(mindist)).to(mindist)
    # Xinits = [X[..., mindist, idxr, :, :] for X in Xopts[1:]]


def split_ms_candidates(
    batch_sizes: List[int], num_fantasies: List[int], Xopt: Tensor
) -> List[Tensor]:
    # this is taken from D16545298
    batch_shape, d = Xopt.shape[:-2], Xopt.shape[-1]
    # X_i needs to have shape f_i x .... x f_1 x batch_shape x q_i x d
    shapes = [
        torch.Size(num_fantasies[:i][::-1] + [*batch_shape, batch_sizes[i], d])
        for i in range(len(num_fantasies) + 1)
    ]
    # Each X_i in Xsplit has shape batch_shape x qtilde x d with
    # qtilde = f_i * ... * f_1 * q_i
    split_sizes = [s[:-3].numel() * s[-2] for s in shapes]
    Xsplit = torch.split(Xopt, split_sizes, dim=-2)
    # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
    perm = [-2] + list(range(len(batch_shape))) + [-1]
    X0 = Xsplit[0].reshape(shapes[0])
    Xother = [
        X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
    ]
    return [X0] + Xother


def merge_sample_runs(a: OneSampleRun, b: OneSampleRun) -> OneSampleRun:
    merged_sample_runs = OneSampleRun(
        algos=a.algos + b.algos,
        func_evals=a.func_evals + b.func_evals,
        recs=a.recs + b.recs,
        train_xs=a.train_xs + b.train_xs,
        train_objs=a.train_objs + b.train_objs,
        train_cons=a.train_cons + b.train_cons,
        train_costs=a.train_costs + b.train_costs,
        wall_times=a.wall_times + b.wall_times,
        acq_value_vec=a.acq_value_vec + b.acq_value_vec,
        cost_evals=a.cost_evals + b.cost_evals,
        failure_mode=a.failure_mode + " + " + b.failure_mode,
        hvs=a.hvs + b.hvs,
        stopping_information=a.stopping_information + b.stopping_information,
        budgets=a.budgets + b.budgets if a.budgets is not None else None,
    )
    return merged_sample_runs


@flow.flow_async()
@flow.typed()
def merge_workflow_results(
    results_list: List[List[OneSampleRun]],
) -> List[OneSampleRun]:
    result = []
    n_trials = len(results_list[0])
    for j in range(n_trials):  # loop over trials
        merged = results_list[0][j]
        for i in range(1, len(results_list)):  # loop over all_packages
            merged = merge_sample_runs(merged, results_list[i][j])
        result.append(merged)
    return result


@flow.flow_async()
@flow.typed()
def convert_state_to_sample_run(bo_state: BOState, algo: str, id: str) -> OneSampleRun:
    return OneSampleRun(
        algos=[id + "-" + algo],
        func_evals=[bo_state.func_evals],
        recs=[bo_state.recs],
        train_xs=[bo_state.train_x],
        train_objs=[bo_state.train_obj],
        train_cons=[[[]]],
        train_costs=[[]],
        wall_times=[[]],
        acq_value_vec=[[]],
        cost_evals=[[]],
        failure_mode="",
        hvs=[[]],
        stopping_information=[[]],
    )


def get_stopping_decision(
    model: Model,
    stopping_rule: str,
    train_x: Tensor,
    train_obj: Tensor,
    num_candidates: int = 1000,
    num_samples: int = 5000,
    stopping_threshold: float = 0.95,
    n_gumbel_samples: int = 100,
) -> Dict[str, Union[bool, float, str]]:
    candidate_set = torch.rand(  # TODO: Should we use Sobol instead?
        num_candidates, train_x.shape[-1], device=train_x.device, dtype=train_x.dtype
    )
    sampler = IIDNormalSampler(num_samples=num_samples, collapse_batch_dims=True)
    if stopping_rule == "probability_optimal":
        stopping_rule = ProbabilityOptimal(
            model=model,
            best_f=train_obj.max(),
            candidate_set=candidate_set,
            sampler=sampler,
            stopping_threshold=stopping_threshold,
        )
    if stopping_rule == "probability_optimal_gumbel":
        stopping_rule = ProbabilityOptimal(
            model=model,
            best_f=train_obj.max(),
            candidate_set=candidate_set,
            sampler=sampler,
            stopping_threshold=stopping_threshold,
            use_gumbel=True,
            n_gumbel_samples=n_gumbel_samples,
        )
    elif stopping_rule == "noisy_probability_optimal":
        stopping_rule = NoisyProbabilityOptimal(
            model=model,
            X_baseline=train_x,
            candidate_set=candidate_set,
            sampler=sampler,
            stopping_threshold=stopping_threshold,
        )
    else:
        raise ValueError(f"Stopping rule {stopping_rule} is unsupported.")

    stopping_information = stopping_rule.evaluate()
    probability_of_optimal = stopping_information["probability_of_optimal"]
    logging.info(f"Probability of optimal = {probability_of_optimal}")
    return stopping_information
