from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tabulate import tabulate

from artemis.experiments import ExperimentFunction
from artemis.general.iteratorize import Iteratorize
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import izip_equal
from artemis.general.test_mode import is_test_mode
from spiking_eqprop.demo_mnist_eqprop import experiment_mnist_eqprop_torch, settings
from spiking_eqprop.quantized_eqprop import EncodingDecodingNeuronLayer

"""
DEPRECATED... Use demo_mnist_quantized_eqprop for with the built-in search instead
"""

experiment_quantized_eqprop = experiment_mnist_eqprop_torch.add_config_root_variant('quantized', layer_constructor = lambda epsilons, quantizer, lambdas=None: EncodingDecodingNeuronLayer.get_simple_constructor(epsilons=epsilons, lambdas=lambdas, quantizer=quantizer))

for network_type, network_settings in settings.items():

    X = experiment_quantized_eqprop.add_root_variant(network_type, **network_settings)
    for eps_function in (0.5, '0.5/sqrt(t)', '0.5/t'):
        X.add_variant(epsilons = eps_function, quantizer='sigma_delta')
        for lambda_function in (0.25, 0.5, 0.8):
            X.add_variant(epsilons = eps_function, lambdas = lambda_function, quantizer='sigma_delta')
    X.add_variant(epsilons='0.89/t**0.43', lambdas='0.89/t**0.43', quantizer='sigma_delta')  # Found in optimal convergence search
    X.add_variant(epsilons='0.843/t**0.0923', lambdas='0.832/t**0.584', quantizer='sigma_delta')  # Found in optimal convergence search


def one_liner(result):
    return f'{len(result["x_iters"])} Runs. ' + ', '.join(f'{k}={v:.3g}' for k, v in izip_equal(result['names'], result['x'])) + f', score = {result["fun"]:.3g}'


def show_record(record):

    result = record.get_result()
    table = tabulate([list(xs)+[fun] for xs, fun in zip(result['x_iters'], result['func_vals'])], headers=list(result['names'])+['score'])
    print(table)


@ExperimentFunction(one_liner_function=one_liner, is_root=True, show=show_record)
def mnist_quantized_eqprop_search(net_version, n_epochs=5, n_calls = 500, seed=1234, n_test_samples=1000, n_random_starts=3, acq_optimizer="auto"):

    rng = get_rng(seed)

    space = [
        Real(0, 1, 'uniform', name='eps_init'),
        Real(0, 1, 'uniform', name='eps_exp'),
        Real(0, 1, 'uniform', name='lambda_init'),
        Real(0, 1, 'uniform', name='lambda_exp'),
    ]

    if is_test_mode():
        n_calls = 3

    @use_named_args(space)
    def objective(eps_init, eps_exp, lambda_init, lambda_exp):
        fixed_params = settings[net_version].copy()
        fixed_params['n_epochs'] = n_epochs
        if is_test_mode():
            fixed_params['n_test_samples'] = 100
            fixed_params['n_epochs'] = 0.5
        else:
            fixed_params['n_test_samples'] = n_test_samples

        result_iterator = experiment_mnist_eqprop_torch(
            layer_constructor = EncodingDecodingNeuronLayer.get_simple_constructor(
                epsilons=f'{eps_init}/t**{eps_exp}',
                lambdas=f'{lambda_init}/t**{lambda_exp}',
                quantizer='sigma_delta',
                ),
            seed = rng,
            **fixed_params
        )
        for result in result_iterator:
            pass
        return result[-1, 'train_error']

    iter = Iteratorize(
        func = lambda callback: gp_minimize(objective,
            dimensions=space,
            n_calls=n_calls,
            n_random_starts = n_random_starts,
            random_state=1234,
            n_jobs=4,
            verbose=False,
            callback=callback,
            acq_optimizer = acq_optimizer,
            ),
    )

    for i, iter_info in enumerate(iter):

        info = dict(names=[s.name for s in space], x_iters =iter_info.x_iters, func_vals=iter_info.func_vals, score = iter_info.func_vals, x=iter_info.x, fun=iter_info.fun)
        latest_info = {s.name: val for s, val in izip_equal(space, iter_info.x_iters[-1])}
        print(f'Latest: {latest_info}, Score: {iter_info.func_vals[-1]:.3g}')
        yield info


for n_epochs in (5, 25):
    mnist_quantized_eqprop_search.add_variant(net_version = 'one_hid', n_epochs=n_epochs)
    mnist_quantized_eqprop_search.add_variant(net_version = 'one_hid_swapless', n_epochs=n_epochs)
    mnist_quantized_eqprop_search.add_variant(net_version = 'big_fast', n_epochs=n_epochs)

mnist_quantized_eqprop_search.add_variant(net_version = 'one_hid', n_epochs=5, acq_optimizer='lbfgs')


"""
Conclusions from these experiments:

- Use one_hid_swapless (swap doesnt work with quantized eqprop, but we knew that already)
- Seems that best parameters found are: 
    net                 n_epochs        Optimization Result
    one_hid_swapless    5               380 Runs. eps_init=0.592, eps_exp=0, lambda_init=0.632, lambda_exp=0,     score = 1.1 
    big_fast            5               61 Runs.  eps_init=0.987, eps_exp=0, lambda_init=1,     lambda_exp=0.818, score = 9
    one_hid_swapless    25              67 Runs.  eps_init=0.726, eps_exp=0, lambda_init=0.614, lambda_exp=0.797, score = 0.3
    big_fast            25              14 Runs.  eps_init=0.86,  eps_exp=0, lambda_init=1,     lambda_exp=0.876, score = 4.2     
    
  <Note- we were using splitstream here, and have since turned it off... So results may differ>

"""



if __name__ == '__main__':

    mnist_quantized_eqprop_search.browse(raise_display_errors=False)
    # mnist_quantized_eqprop_search.get_variant('one_hid').run()


    # eps_init=0.822, eps_exp=0.367, lambda_init=0.18, lambda_exp=0.17, score = 6.8
    # experiment_quantized_eqprop.get_variant('one_hid').add_variant(epsilons='0.822/t**0.367', lambdas='0.17/t**0.17', quantizer='sigma_delta').run()
    # experiment_quantized_eqprop.get_variant('one_hid_swapless').add_variant(epsilons='0.822/t**0.367', lambdas='0.17/t**0.17', quantizer='sigma_delta').run()


