from typing import Sequence, Union, Tuple, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt

from artemis.experiments import ExperimentFunction
from artemis.experiments.experiment_record import ExperimentRecord
from artemis.experiments.experiment_record_view import separate_common_args
from artemis.general.checkpoint_counter import Checkpoints
from artemis.general.display import dict_to_str
from artemis.general.duck import Duck
from artemis.general.numpy_helpers import get_rng
from artemis.general.progress_indicator import ProgressIndicator
from artemis.general.speedometer import Speedometer
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.tools.costs import percent_argmax_incorrect
from artemis.ml.tools.iteration import minibatch_index_info_generator
from artemis.ml.tools.running_averages import RunningAverage
from artemis.plotting.expanding_subplots import select_subplot
from artemis.plotting.pyplot_plus import get_lines_color_cycle
from spiking_eqprop.eq_prop import SimpleLayerController, run_inference, \
    initialize_states, initialize_params, run_eqprop_training_update, output_from_state, LayerParams, IDynamicLayer

"""
Re-implmentation of equilibrium propagation with a functional neuron interface.  
"""


def report_score_from_result(result):
    epoch, test_error, train_error = result[-1, 'epoch'], result[-1, 'test_error'], result[-1, 'train_error']
    return f'Epoch: {epoch:.3g}, Test Error: {test_error:.3g}%, Train Error: {train_error:.3g}'


def compare(records: Sequence[ExperimentRecord], show_now = True):

    argcommon, argdiffs = separate_common_args(records, as_dicts=True, only_shared_argdiffs=False)

    ax = select_subplot(1)
    color_cycle = get_lines_color_cycle()
    for i, (rec, ad, c) in enumerate(zip(records, argdiffs, color_cycle)):
        result = rec.get_result()

        for i, subset in enumerate(('train_error', 'test_error')):

            is_train = subset=="train_error"

            ax.plot(result[:, 'epoch'], result[:, subset], label=dict_to_str(ad).replace('lambdas', '$\lambda$').replace('epsilon', '$\epsilon$'), linestyle='--' if is_train else '-', alpha=0.7 if is_train else 1, color=c)
            ax.grid()
            ax.legend()
            ax.set_ybound(0, max(10, min(result[:, subset])*1.5))
            # ax.set_ylabel(f'{# "Train" if is_train else "Test"} % Error')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Error')
    ax.grid()
    # plt.legend([f'{alg}{subset}' for alg in ['Real Eq-Prop', f'Bin Eq-Prop $\lambda$={records[-1].get_args()["lambdas"]}'] for subset in ['Train', 'Test']])
    # plt.ion()
    if show_now:
        plt.show()
    print('ENter')
    # import IPython; IPython.embed()


PLOT_MODE = True


def percent_argmax_incorrect(prediction, y):

    pred_ixs = torch.argmax(prediction, dim=1)
    true_ixs = torch.argmax(y, dim=1)
    return 100*torch.mean((pred_ixs!=true_ixs).float())


USE_CUDA_WHEN_AVAILABLE = True


def set_use_cuda(state):
    global USE_CUDA_WHEN_AVAILABLE
    USE_CUDA_WHEN_AVAILABLE = state


@ExperimentFunction(is_root=True, one_liner_function=report_score_from_result, compare=compare)
def experiment_mnist_eqprop_torch(
        layer_constructor: Callable[[int, LayerParams], IDynamicLayer],
        n_epochs = 10,
        hidden_sizes = (500, ),
        minibatch_size = 20,
        beta = .5,
        random_flip_beta = True,
        learning_rate = .05,
        n_negative_steps = 20,
        n_positive_steps = 4,
        initial_weight_scale = 1.,
        online_checkpoints_period = None,
        epoch_checkpoint_period = {0: .25, 1: .5, 5: 1, 10: 2, 50: 4},
        skip_zero_epoch_test = False,
        n_test_samples = 10000,
        prop_direction: Union[str, Tuple]='neutral',
        bidirectional = True,
        renew_activations = True,
        do_fast_forward_pass = False,
        rebuild_coders = True,
        l2_loss = None,
        splitstream = False,
        seed = 1234,
        ):
    """
    Replicate the results of Scellier & Bengio:
        Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation
        https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full

    Specifically, the train_model demo here:
        https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop

    Differences between our code and theirs:
    - We do not keep persistent layer activations tied to data points over epochs.  So our results should only really match for the first epoch.
    - We evaluate training score periodically, rather than online average (however you can see online score by setting online_checkpoints_period to something that is not None)
    """
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() and USE_CUDA_WHEN_AVAILABLE else 'cpu'
    if device=='cuda':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f'Using Device: {device}')

    print('Params:\n' + '\n'.join(list(f'  {k} = {v}' for k, v in locals().items())))

    rng = get_rng(seed)
    n_in = 784
    n_out = 10
    dataset = get_mnist_dataset(flat=True, n_test_samples=None).to_onehot()
    x_train, y_train = (torch.tensor(a.astype(np.float32)).to(device) for a in dataset.training_set.xy)
    x_test, y_test = (torch.tensor(a.astype(np.float32)).to(device) for a in dataset.test_set.xy)  # Their 'validation set' is our 'test set'
    x_val, y_val = (torch.tensor(a.astype(np.float32)).to(device) for a in dataset.validation_set.xy)  # Their 'validation set' is our 'test set'

    if is_test_mode():
        x_train, y_train, x_test, y_test, x_val, y_val = x_train[:100], y_train[:100], x_test[:100], y_test[:100], x_val[:100], y_val[:100]
        n_epochs=1
        n_negative_steps = 3
        n_positive_steps = 3

    layer_sizes = [n_in] + list(hidden_sizes) + [n_out]

    # y_train = y_train.astype(np.float32)

    ra = RunningAverage()
    sp = Speedometer(mode='last')
    is_online_checkpoint = Checkpoints(online_checkpoints_period, skip_first=skip_zero_epoch_test) if online_checkpoints_period is not None else lambda: False
    is_epoch_checkpoint = Checkpoints(epoch_checkpoint_period, skip_first=skip_zero_epoch_test)

    training_states = initialize_states(
        layer_constructor=layer_constructor,
        n_samples=minibatch_size,
        params = initialize_params(
            layer_sizes=layer_sizes,
            initial_weight_scale=initial_weight_scale,
            rng=rng
            )
        )

    # dbplot(training_states[0].params.w_fore[:10, :10], str(rng.randint(265)))

    if isinstance(prop_direction, str):
        fwd_prop_direction, backward_prop_direction = prop_direction, prop_direction
    else:
        fwd_prop_direction, backward_prop_direction = prop_direction

    def do_test():
        # n_samples = n_test_samples if n_test_samples is not None else len(x_test)
        test_error, train_error, val_error = [
            percent_argmax_incorrect(
                run_inference(
                    x_data=x[:n_test_samples],
                    states=initialize_states(
                        layer_constructor=layer_constructor,
                        params=[s.params for s in training_states],
                        n_samples=n_samples
                        ),
                    n_steps=n_negative_steps,
                    prop_direction=fwd_prop_direction,
                    ),
                y[:n_samples]).item()
            for x, y in [(x_test, y_test), (x_train, y_train), (x_val, y_val)]
            for n_samples in [min(len(x), n_test_samples) if n_test_samples is not None else len(x)]]  # Not an actal loop... just hack for assignment in comprehensions
        print(f'Epoch: {epoch:.3g}, Iter: {i}, Test Error: {test_error:.3g}%, Train Error: {train_error:.3g}, Validation Error: {val_error:.3g}, Mean Rate: {sp(i):.3g}iter/s')

        return dict(iter=i, epoch=epoch, train_error=train_error, test_error=test_error, val_error=val_error)

    results = Duck()
    pi = ProgressIndicator(expected_iterations=n_epochs*dataset.training_set.n_samples/minibatch_size, update_every='10s')
    for i, (ixs, info) in enumerate(minibatch_index_info_generator(n_samples=x_train.size()[0], minibatch_size=minibatch_size, n_epochs=n_epochs)):
        epoch = i*minibatch_size/x_train.shape[0]

        if is_epoch_checkpoint(epoch):
            with pi.pause_measurement():
                results[next, :] = do_test()
                yield results
                if epoch>2 and results[-1, 'train_error']>50:
                    return

        # The Original training loop, just taken out here:
        x_data_sample, y_data_sample = x_train[ixs], y_train[ixs]
        training_states = run_eqprop_training_update(x_data=x_data_sample, y_data=y_data_sample, layer_states = training_states, beta=beta, random_flip_beta=random_flip_beta,
                                                     learning_rate=learning_rate, layer_constructor=layer_constructor, bidirectional=bidirectional, l2_loss=l2_loss, renew_activations=renew_activations,
                                                     n_negative_steps=n_negative_steps, n_positive_steps=n_positive_steps, prop_direction=prop_direction, splitstream=splitstream, rng=rng)
        this_train_score = ra(percent_argmax_incorrect(output_from_state(training_states), y_train[ixs]))
        if is_online_checkpoint():
            print(f'Epoch {epoch:.3g}: Iter {i}: Score {this_train_score:.3g}%: Mean Rate: {sp(i):.2g}')

        pi.print_update(info=f'Epoch: {epoch}')

    results[next, :] = do_test()
    yield results



settings = dict(
    # one_hid = dict(
    #     hidden_sizes = [500],
    #     learning_rate = [0.1, 0.05],
    #     n_epochs=25,
    #     n_negative_steps=20,
    #     n_positive_steps=1,
    #     prop_direction='swap',
    #     ),
    one_hid_swapless = dict(
        hidden_sizes = [500],
        learning_rate = [0.1, 0.05],
        n_epochs=25,
        n_negative_steps=20,
        n_positive_steps=4,
        prop_direction='neutral',
        ),
    # two_hid = dict(
    #     hidden_sizes = [500, 500],
    #     learning_rate = [0.4, 0.1, 0.008],
    #     n_epochs=50,
    #     n_negative_steps=60,
    #     n_positive_steps=1,
    #     prop_direction='swap',
    # ),
    big_fast = dict(
        hidden_sizes = [500, 500, 500],
        n_epochs = 500,
        minibatch_size = 20,
        n_negative_steps = 50,
        n_positive_steps = 8,
        beta = 1.,
        learning_rate = [.128, .032, .008, .002],
        skip_zero_epoch_test = True
    )
)
# ======================================================================================================================


X_1hid= experiment_mnist_eqprop_torch.add_root_variant('1_hid', **settings['one_hid_swapless'])
X_3hid = experiment_mnist_eqprop_torch.add_root_variant('3_hid', **settings['big_fast'])


X_1hid_vanilla, X_3hid_vanilla = (X.add_config_variant('vanilla', layer_constructor = lambda epsilon=0.5: SimpleLayerController.get_partial_constructor(epsilon=epsilon)) for X in (X_1hid, X_3hid))

X_1hid_vanilla.add_variant(splitstream=True)

"""
Does it work?
- Yes
"""

# ======================================================================================================================

# experiment_plain_old_eqprop.add_variant(bidirectional = False)  # Kill these once we have he next results
# experiment_plain_old_eqprop.add_variant(bidirectional = 'full')
"""
Is bidirectional better?
- Doesn't seem to make much difference.  
"""
X_1hid_vanilla.add_variant(bidirectional = False)
X_1hid_vanilla.add_variant(bidirectional = 'full')

# ======================================================================================================================



if __name__ == '__main__':
    # X_1hid_vanilla.call()
    experiment_mnist_eqprop_torch.browse(catch_errors=False)
    # Xlarge.call()

