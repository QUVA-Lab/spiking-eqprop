import numpy as np
from matplotlib import pyplot as plt

from artemis.general.duck import Duck
from artemis.plotting.expanding_subplots import subplot_at
from spiking_eqprop.quantized_eqprop import ScheduledStepSizer, KestonsStepSizer, OptimalStepSizer


def demo_stochastic_approximator_tracking(step_sizer_dict, n_steps = 100, tc=5, noise=2, mag=10, seed = None):

    t = np.arange(n_steps)

    true_signals = {
        'approach': mag*(1-np.exp(-t/tc)),
        'delayed_sweep': mag/(1+np.exp(-(t-n_steps/2)/tc)),
        'step': mag*(t>(n_steps/2)),
        'bump': mag*((t>(n_steps/3))&(t<2*n_steps/3)),
    }

    # x_1_true = mag*(1-np.exp(-t/tc))
    # x_2_true = mag/(1+np.exp(-(t-n_steps/2)/tc))

    rng = np.random.RandomState(seed)

    noise = noise*rng.randn(n_steps)

    # x_1_noisy = x_1_true + noise
    # x_2_noisy = x_2_true + noise

    results = Duck()
    fig = plt.figure(figsize=(14, 8))
    # for i, (x_sig, x_true) in enumerate(zip([x_1_noisy, x_2_noisy], [x_1_true, x_2_true])):
    for i, (name, x_true) in enumerate(true_signals.items()):

        x_sig = x_true + noise

        ax_sig = subplot_at(i, 0, fig=fig)
        ax_err = subplot_at(i, 1, fig=fig)
        ax_step = subplot_at(i, 2, fig=fig)
        ax_sig.grid()
        ax_err.grid()
        ax_step.grid()

        ax_sig.plot(x_true, color='k', linewidth=2)
        ax_sig.plot(x_sig, color='gray', linewidth=2)

        for filtername, step_sizer in step_sizer_dict.items():
            average_signal, step_sizes = zip(*[(avg, a) for avg in [0] for st in [step_sizer] for x in x_sig for st, a in [st(x)] for avg in [(1-a) * avg + a * x]])
            h, = ax_sig.plot(average_signal, label=filtername)
            ax_sig.set_ylabel('Signal')
            error = np.abs(np.array(average_signal) - x_true)

            results[i, filtername, 'rmse'] = np.sqrt(np.mean(error**2))

            ax_err.plot(error, color=h.get_color())
            ax_err.set_ylabel('Error')

            ax_step.plot(step_sizes, color=h.get_color(), label=filtername)
            ax_step.set_ylabel('Step Size')
        ax_step.legend()

    print(results.description())

    plt.show()


if __name__ == '__main__':
    demo_stochastic_approximator_tracking(
        step_sizer_dict={
            'simple': ScheduledStepSizer('1/t'),
            'sqrt': ScheduledStepSizer('1/sqrt(t)'),
            'burnin-simple': ScheduledStepSizer('5/(5+t)'),
            'osa': OptimalStepSizer(error_stepsize_target=0.1),
            'kestons': KestonsStepSizer(a=5, b=5),
            # 'miroz': MirozamedovStepSizer(a=1., delta=0),
            # 'osa': OptimalStepSizer(error_stepsize_target=0.1),
        }

    )