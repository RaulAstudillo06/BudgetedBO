from botorch import acquisition
import random
import torch

from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead, warmstart_multistep
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import IIDNormalSampler
from torch import Tensor
from typing import List, Dict



def custom_warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    full_optimizer: Tensor,
    algo_params: Dict,
) -> Tensor:

    batch_initial_conditions = warmstart_multistep(
            acq_function=acq_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            full_optimizer=full_optimizer,
        )
    
    n_initial_points = batch_initial_conditions.shape[0]
    random_index = random.randrange(n_initial_points)
    input_dim = batch_initial_conditions.shape[-1]
    batch_shape, shapes, sizes = acq_function.get_split_shapes(X=batch_initial_conditions)

    # Copy of objective model
    obj_model = acq_function.model.subset_output(idcs=[0])

    #Define optimization domain
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    i = 0

    for size in sizes:
        # Auxiliar acquisition fucntion
        aux_acq_func_id = "PM"
        if aux_acq_func_id == "PM":
            aux_acq_func = PosteriorMean(model=obj_model)

        elif aux_acq_func_id == "EI":
            standardized_obj_vals = obj_model.train_targets
            obj_vals = obj_model.outcome_transform.untransform(standardized_obj_vals)[0]
            best_f = torch.max(obj_vals).item()

            aux_acq_func = ExpectedImprovement(model=obj_model, best_f=best_f)

        # Get new point
        new_x, acq_value = optimize_acqf(
            acq_function=aux_acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=5 * input_dim,
            raw_samples=100 * input_dim,
            options={
                "batch_limit": 5,
                "maxiter": 100,
                "method": "L-BFGS-B",
            },
            return_best_only=True,
        )

        print(new_x)
        for _ in range(size):
            batch_initial_conditions[random_index, i, :] = new_x.clone().squeeze(0)
            i += 1

        # Fantasize objective and cost values
        obj_sampler = IIDNormalSampler(
            num_samples=1, resample=True, collapse_batch_dims=True
        )
        obj_post_X = obj_model.posterior(new_x, observation_noise=True)
        fantasy_obj_val = obj_sampler(obj_post_X).squeeze(dim=0)
        obj_model = obj_model.condition_on_observations(X=new_x, Y=fantasy_obj_val.detach())

    print(batch_initial_conditions[random_index])
    return batch_initial_conditions
    