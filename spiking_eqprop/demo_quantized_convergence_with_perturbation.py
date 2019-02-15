from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from skopt.space import Real

from artemis.experiments import ExperimentFunction
from artemis.experiments.experiment_record import ExperimentRecord
from artemis.experiments.experiment_record_view import separate_common_args
from artemis.general.checkpoint_counter import do_every
from artemis.general.mymath import cosine_distance
from artemis.general.numpy_helpers import get_rng
from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.should_be_builtins import izip_equal
from artemis.general.test_mode import is_test_mode
from artemis.plotting.data_conversion import put_vector_in_grid
from artemis.plotting.db_plotting import dbplot, dbplot_redraw_all
from artemis.plotting.easy_plotting import funplot
from artemis.plotting.matplotlib_backend import LinePlot
from artemis.plotting.pyplot_plus import outside_right_legend
from spiking_eqprop.eq_prop import initialize_states, initialize_params, eqprop_step, \
    SimpleLayerController
from spiking_eqprop.pytorch_helpers import to_default_tensor
from spiking_eqprop.quantized_eqprop import EncodingDecodingNeuronLayer, PredictiveEncoder, \
    create_step_sizer, PredictiveDecoder, create_quantizer, IdentityFunction, OptimalStepSizer
import torch


"""
Ok, so this demonstrates how we can schedule the epsilons and lambdas of the predictive coder to guarantee convergence.  

Long story short: 1/t annealing is too fast - Early values stay remembered and network does not converge to fixed point.
Same with exponential scaling.  With no annealing, we hit a noise-floor quickly.  In general 1/t**k annealing seems to 
work when k>0<1 with 1/sqrt(t) working pretty well for both epsilons and lambdas.  

Annealing lambdas seems to lead to faster convergence than annealing epsilons.  
"""


def rho(x):
    return np.clip(x, 0, 1)


def drho(x):
    return ((x>=0) & (x<=1)).astype(float)


def last(iterable):
    gen = iter(iterable)
    x = next(gen)
    for x in gen:
        pass


def compare_quantized_convergence_records(records: Sequence[ExperimentRecord], add_reference_lines = True, label_reference_lines = True, ax=None, show_now = True, include_legend_now = True, legend_squeeze=0.5):

    plt.figure()
    argcommon, argdiffs = separate_common_args(records, as_dicts=True, only_shared_argdiffs=True)
    if ax is None:
        # plt.figure(figsize=(10, 6))
        ax = plt.gca()

    for record_ix, (rec, args) in enumerate(zip(records, (dict(d) for d in argdiffs))):
        distance_to_converged, smooth_endpoint_delta, rough_endpoint_delta = rec.get_result()
        # argstr = str(args)

        # h, = plt.semilogy(np.arange(1, len(distance_to_converged)+1), distance_to_converged.mean(axis=1), label=rec.get_experiment().name[rec.get_experiment().name.index('.'):])
        h, = plt.semilogy(np.arange(1, len(distance_to_converged)+1), distance_to_converged.mean(axis=1), label=str(args))

        all_args = rec.get_args()
        perturbation_step = int(all_args['n_steps'] * all_args['change_frac'])
        plt.axvline(perturbation_step, color='k', linestyle = '--', linewidth=2)
        ax.set_ylabel('$\\frac{1}{|S|}\sum_{i \in S}\left|s_i^t - s_i^{fixed}\\right|$')
        ax.grid(True)

        if record_ix==len(records)-1:
            if add_reference_lines:
                arbitrary_base = .3
                funplot(lambda t: arbitrary_base/np.sqrt(t), color='k', label='$\propto 1/\sqrt{t}$' if label_reference_lines else None, linestyle = ':', keep_ylims=True)
                funplot(lambda t: arbitrary_base/t, color='k', label='$\propto 1/t$' if label_reference_lines else None, linestyle = '--', keep_ylims=True)
            if include_legend_now:
                outside_right_legend(width_squeeze=legend_squeeze, ax=ax)

    if show_now:
        plt.show()


def show_error(results):
    errors, smooth_endpoint_delta, rough_endpoint_delta = results
    cd = cosine_distance(smooth_endpoint_delta, rough_endpoint_delta, axis=1).mean()
    return f'Mean: {np.mean(errors):6.3g},\tFinal: {np.mean(errors[-1]):.3g}, Delta-Diff: {cd:.3g}'


@ExperimentFunction(is_root=True, one_liner_function=show_error, compare=compare_quantized_convergence_records)
def demo_quantized_convergence_perturbed(
        quantized_layer_constructor,
        smooth_epsilon=0.5,
        layer_sizes=(500, 500, 10),
        initialize_acts_randomly = False,
        minibatch_size = 20,
        # n_steps = 100,
        smooth_longer_factor = 10,
        n_steps = 500,
        change_frac = 0.5,
        beta = .5,  # TODO: revert
        initial_weight_scale = 1.,
        prop_direction = 'neutral',
        data_seed=None,
        param_seed = None,
        hang = True,
        ):
    """
    """

    perturbation_step = int(n_steps * change_frac)
    smooth_layer_constructor = SimpleLayerController.get_partial_constructor(epsilon=smooth_epsilon)

    print('Params:\n' + '\n'.join(list(f'  {k} = {v}' for k, v in locals().items())))

    data_rng = get_rng(data_seed)
    param_rng = get_rng(param_seed)

    HISTORY_LEN = n_steps
    N_NEURONS_TO_PLOT = 10

    if is_test_mode():
        n_steps = 10
        perturbation_step = 5

    pi = ProgressIndicator(update_every='2s', expected_iterations=2*n_steps)
    n_in, n_out = layer_sizes[0], layer_sizes[-1]

    x_data = to_default_tensor(data_rng.rand(minibatch_size, n_in))

    y_data = torch.zeros((minibatch_size, n_out))
    y_data[np.arange(len(y_data)), np.random.choice(n_out, size=minibatch_size)] = 1

    params = initialize_params(
        layer_sizes=layer_sizes,
        initial_weight_scale=initial_weight_scale,
        rng=param_rng
        )

    def run_update(layer_constructor, mode):

        if PLOT:
            plt.gca().set_color_cycle(None)

        states = initialize_states(
            layer_constructor=layer_constructor,
            n_samples=minibatch_size,
            params = params
            )

        for t in range(n_steps):

            for _ in range(smooth_longer_factor) if mode=='Smooth' else range(1):
                if t < perturbation_step:
                    states = eqprop_step(layer_states=states, x_data=x_data, beta=0, y_data=None, direction=prop_direction)
                else:
                    states = eqprop_step(layer_states=states, x_data=x_data, beta=beta, y_data=y_data, direction=prop_direction)

            acts = [s.potential for s in states]
            yield acts[1:]
            if PLOT:
                if do_every('2s'):
                    dbplot([put_vector_in_grid(a[0]) for a in acts], f'acts-{mode}', title=f'{mode} Iter-{t}')
                if mode=='Rough':
                    dbplot([states[1].stepper.step_size.mean(), states[2].stepper.step_size.mean()], 'step size', draw_every='2s')
            pi()

        if PLOT:
            dbplot_redraw_all()

    rough_record = list(run_update(layer_constructor=quantized_layer_constructor, mode='Rough'))

    smooth_record = list(run_update(layer_constructor=smooth_layer_constructor, mode='Smooth'))

    smooth_neg_endpoint = smooth_record[perturbation_step-1]
    smooth_pos_endpoint = smooth_record[-1]

    smooth_endpoint_delta = np.concatenate([sp-sn for sp, sn in izip_equal(smooth_pos_endpoint, smooth_neg_endpoint)], axis=1)

    rough_endpoint_delta = np.concatenate([sp-sn for sp, sn in izip_equal(rough_record[-1], rough_record[perturbation_step-1])], axis=1)

    distance_to_converged = np.array(
        [[torch.mean(abs(hr - hs)).item() for hr, hs in zip(hs_rough, smooth_neg_endpoint)] for hs_rough in rough_record[:perturbation_step]]+
        [[torch.mean(abs(hr - hs)).item() for hr, hs in zip(hs_rough, smooth_pos_endpoint)] for hs_rough in rough_record[perturbation_step:]]
    )  # (n_steps, n_layers) array indicating convergence to the fixed point for each non-input layer.

    # plt.semilogy(distance_to_converged)
    # plt.show()

    if PLOT:
        dbplot(distance_to_converged, 'errors', plot_type=lambda: LinePlot(x_axis_type='log', y_axis_type='log'), legend=[f'Layer {i+1}' for i in range(len(layer_sizes))])

    mean_abs_error = np.mean(distance_to_converged, axis=0)
    final_abs_error = distance_to_converged[-1]
    print(f'Mean Abs Layerwise Errors: {np.array_str(mean_abs_error, precision=5)}\t Final Layerwise Errors: {np.array_str(final_abs_error,  precision=5)}')

    return distance_to_converged, rough_endpoint_delta, smooth_endpoint_delta


X_scheduled = demo_quantized_convergence_perturbed.add_config_root_variant('scheduled', quantized_layer_constructor = lambda epsilons, lambdas=None, quantizer='sigma_delta': EncodingDecodingNeuronLayer.get_simple_constructor(epsilons=epsilons, lambdas=lambdas, quantizer=quantizer))


X_quanttype = X_scheduled.add_root_variant('quantization_type')
for epsilons in ('1/sqrt(t)', '1/t'):
    for quantizer in ('sigma_delta', 'stochastic', 'threshold', 'second_order_sd'):
        X_quanttype.add_variant(epsilons =epsilons, quantizer=quantizer)


# ======================================================================================
# Try Adaptive


X_nonadaptive = demo_quantized_convergence_perturbed.add_config_root_variant('nonadaptive', quantized_layer_constructor = lambda epsilons, lambdas=None, quantizer='sigma_delta':
        EncodingDecodingNeuronLayer.get_simple_constructor(
            epsilons=epsilons,
            lambdas=lambdas,
            quantizer=quantizer)
            )

X_quant = X_nonadaptive.add_root_variant(epsilons = '1/t**.75', lambdas='.5/t**.5')
X_quant.add_variant(quantizer='sigma_delta')
X_quant.add_variant(quantizer='stochastic')

X_nonadaptive.add_variant(epsilons=0.5)
X_nonadaptive.add_variant(epsilons='1/sqrt(t)')  # This goes to zero
X_nonadaptive.add_variant(epsilons='1/t')  # As expected, this don't be going to zero

# Create a variant with a polynomial annealing schedule on both epsilon and lambda
X_scheduled = X_nonadaptive.add_config_root_variant('poly_schedule',
    epsilons = lambda eps_init, eps_exp: f'{eps_init}/t**{eps_exp}', lambdas = lambda lambda_init, lambda_exp: f'{lambda_init}/t**{lambda_exp}')
X_scheduled.add_parameter_search(
    fixed_args = dict(param_seed=None, data_seed=None),
    space = dict(eps_init = Real(0, 1, 'uniform'), eps_exp = Real(0, 1, 'uniform'), lambda_init = Real(0, 1, 'uniform'), lambda_exp = Real(0, 1, 'uniform')),
    # scalar_func=lambda result: result[0].mean(),
    scalar_func=lambda result: result[0][[249, -1]].mean(),
    n_calls=500
    )
# X_scheduled.add_variant('gp_params', eps_init=0.506, eps_exp=0, lambda_init=0.643, lambda_exp=0.471)
X_scheduled.add_variant('gp_params', eps_init=0.758, eps_exp=0.0548, lambda_init=0.892, lambda_exp=0.556)
X_scheduled.add_variant('gp_params_end', eps_init=0.775, eps_exp=0.313, lambda_init=0.67, lambda_exp=0.578)

# Create a variant with a polynomial annealing schedule, just on epsilon
X_scheduled_nolambda = X_scheduled.add_root_variant(lambda_init=1, lambda_exp=0)
X_scheduled_nolambda.add_parameter_search(
    fixed_args = dict(param_seed=None, data_seed=None),
    space = dict(eps_init = Real(0, 1, 'uniform'), eps_exp = Real(0, 1, 'uniform')),
    # scalar_func=lambda result: result[0].mean(),
    scalar_func=lambda result: result[0][[249, -1]].mean(),
    n_calls=500
    )
# X_scheduled_nolambda.add_variant('gp_params', eps_init=0.969, eps_exp=0.455).add_variant(quantizer='stochastic')
# X_scheduled_nolambda.add_variant('gp_params', eps_init=0.969, eps_exp=0.455).add_variant(quantizer='stochastic')
X_scheduled_nolambda.add_variant('gp_params', eps_init=0.529, eps_exp=0.35)
X_scheduled_nolambda.add_variant('gp_params_end', eps_init=0.857, eps_exp=0.534)


X_adaptive_baseline = demo_quantized_convergence_perturbed.add_config_root_variant('adaptive',
     quantized_layer_constructor = lambda stepper, quantizer='sigma_delta', lambdas=None:
          EncodingDecodingNeuronLayer.get_partial_constructor(
              encoder = create_quantizer(quantizer) if lambdas is None else PredictiveEncoder(create_step_sizer(lambdas), quantizer=create_quantizer(quantizer)),
              decoder = IdentityFunction() if lambdas is None else PredictiveDecoder(create_step_sizer(lambdas)),
              stepper = stepper
              )
      )


# X_kestens = X_adaptive_baseline.add_config_variant('kestons', stepper = lambda a=10, b=10: KestonsStepSizer(a=a, b=b))
# X_kestens.add_root_variant(param_seed=None, data_seed=None).add_parameter_search(
#     space=dict(a = Real(1, 100, 'log-uniform'), b = Real(1, 100, 'log-uniform')),
#     scalar_func=lambda result: result[0].mean()
#     )
# # X_kestens.add_variant('gp_params', a=9.83, b=9.7)
# X_kestens.add_variant('gp_params', a=6.81, b=2.64)
# X_kestens.add_variant(quantizer = 'stochastic')


# X_miroz = X_adaptive_baseline.add_config_variant('miroz', stepper = lambda a=1, delta=0: MirozamedovStepSizer(a=a, delta=delta))
# X_miroz.add_variant(delta = .1)

# Create a variant using OSA for the learning-rate step-size
X_optimal = X_adaptive_baseline.add_config_variant('optimal', stepper = lambda error_stepsize_target = .01, epsilon=1e-7: OptimalStepSizer(error_stepsize_target=error_stepsize_target, epsilon=epsilon))

# Create a parameter search for the optimal OSA parameter
X_optimal.add_parameter_search(
        fixed_args=dict(param_seed=None, data_seed=None),
        space = dict(error_stepsize_target = Real(0.001, 1, 'log-uniform')),
        # scalar_func=lambda result: result[0].mean(),
        scalar_func=lambda result: result[0][[249, -1]].mean(),
        n_calls=500
        )
# X_optimal.add_variant('gp_params', error_stepsize_target=0.096)
X_optimal.add_variant('gp_params', error_stepsize_target=0.0595)
X_optimal.add_variant('gp_params_end', error_stepsize_target=0.0503)

# Create a variant of the last one where we also search over predictive coding coefficients
X_optimal.add_parameter_search('parameter_search_predictive',
        fixed_args = dict(param_seed=None, data_seed=None),
        space = dict(error_stepsize_target = Real(0.001, 1, 'log-uniform'), lambdas = Real(0, 1, 'uniform')),
        # scalar_func=lambda result: result[0].mean(),
        scalar_func=lambda result: result[0][[249, -1]].mean(),
        n_calls=500
        )
X_optimal.add_variant('gp_params_predictive', error_stepsize_target=0.132, lambdas=0.156)
X_optimal.add_variant('gp_params_predictive_end', error_stepsize_target=0.0411, lambdas=0.206)
# X_optimal.add_variant('gp_params_predictive', error_stepsize_target=0.125, lambdas=0.531)
# X_optimal.add_variant(error_stepsize_target = 0.1)
# X_optimal.add_variant(error_stepsize_target = 0.001)
# X_optimal.add_variant(error_stepsize_target = 0.5)
# X_optimal.add_variant(epsilon=0.1)
# X_optimal.add_variant(quantizer = 'stochastic')
# X_optimal.add_variant(prop_direction='forward')


# Is it possible to use an adaptive step size to achieve as good or better convergence than any fixed schedule?
# - Currently, no, looks like fixed schedule does best, but lets try OSA AND lambda-schedule
X_osa_lamba_scheduled_baseline = X_optimal.add_config_root_variant('OSA-lambda_sched', lambdas = lambda lambda_init, lambda_exp: f'{lambda_init}/t**{lambda_exp}')
X_osa_lamba_scheduled_baseline.add_parameter_search('parameter_search_predictive',
        fixed_args=dict(param_seed=None, data_seed=None),
        space = dict(error_stepsize_target = Real(0.001, 1, 'log-uniform'), lambda_init = Real(0, 1, 'uniform'), lambda_exp = Real(0, 1, 'uniform')),
        # scalar_func=lambda result: result[0].mean(),
        scalar_func=lambda result: result[0][[249, -1]].mean(),
        n_calls=500
        )
# X_osa_lamba_scheduled_baseline.add_variant('gp_params', error_stepsize_target=0.362, lambda_init=0.521, lambda_exp=0.383)
X_osa_lamba_scheduled_baseline.add_variant('gp_params', error_stepsize_target=0.392, lambda_init=0.906, lambda_exp=0.439)
X_osa_lamba_scheduled_baseline.add_variant('gp_params_end', error_stepsize_target=0.108, lambda_init=0.39, lambda_exp=0.45)


X_doubly_adaptive_baseline = demo_quantized_convergence_perturbed.add_config_root_variant('double_trouble', quantized_layer_constructor = lambda epsilons, lambdas=None, dec_lambdas=None, quantizer='sigma_delta':
        EncodingDecodingNeuronLayer.get_simple_constructor(
            epsilons=epsilons,
            lambdas=lambdas,
            dec_lambdas = dec_lambdas,
            quantizer=quantizer,
            use_direct_encoder=False
        )
    )


PLOT = False

if __name__ == '__main__':
    # demo_quantized_convergence_perturbed.browse(display_format='flat', remove_prefix=False)
    demo_quantized_convergence_perturbed.browse(display_format='flat', remove_prefix=False, filterexp='tag:psearch')
    # record = X_optimal.add_variant('test', error_stepsize_target=0.2, lambdas=0.156).run()
    # compare_quantized_convergence_records([record])