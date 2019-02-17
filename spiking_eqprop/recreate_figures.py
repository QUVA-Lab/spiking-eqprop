from matplotlib import pyplot as plt

from artemis.experiments import load_experiment
from artemis.experiments.experiment_management import get_multiple_records
from artemis.experiments.experiment_record_view import plot_hyperparameter_search
from artemis.experiments.experiments import Experiment
from artemis.plotting.expanding_subplots import set_figure_border_size, add_subplot
from artemis.plotting.pyplot_plus import outside_right_legend
from artemis.plotting.range_plots import plot_sample_mean_and_var
from spiking_eqprop.demo_quantized_convergence_with_perturbation import demo_quantized_convergence_perturbed
from spiking_eqprop.demo_mnist_eqprop import report_score_from_result, X_1hid_vanilla
from spiking_eqprop.demo_mnist_quantized_eqprop import experiment_mnist_eqprop_torch, X_polyscheduled_longer, \
    X_osa_longer_1hid_psearch, X_osa_longer_gp_params
from spiking_eqprop.demo_signal_figure import demo_create_signal_figure


def create_neuron_figure():

    demo_create_signal_figure()


def create_convergence_var_figure(n=20):

    for v in demo_quantized_convergence_perturbed.get_all_variants():
        print(v.name)

    experiments = [
        ('$\epsilon = 1/\sqrt{t}$', 'demo_quantized_convergence_perturbed.nonadaptive.epsilons=1_SLASH_sqrt(t)'),
        ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.nonadaptive.poly_schedule.gp_params_end'),
        ('$\epsilon$ = OSA($\\bar\\nu$)', 'demo_quantized_convergence_perturbed.adaptive.optimal.gp_params_end'),
        ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', 'demo_quantized_convergence_perturbed.adaptive.optimal.gp_params_predictive_end'),
        ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.adaptive.optimal.OSA-lambda_sched.gp_params_end'),
    ]

    ax = plt.gca()
    ax.set_yscale('log')
    for name, exp_id in experiments:

        exp = load_experiment(exp_id)
        records = get_multiple_records(exp_id, n=n, only_completed=True, if_not_enough='run')
        err = [errors.mean(axis=1) for r in records for errors, _, _ in [r.get_result()]]

        plot_sample_mean_and_var(err, label=name, fill_alpha=0.15)

        all_args = exp.get_args()
        perturbation_step = int(all_args['n_steps'] * all_args['change_frac'])-1
    plt.axvline(perturbation_step, color='k', linestyle = '--', linewidth=2)
    plt.ylabel('$\\frac{1}{|S|}\sum_{i \in S}\left|s_i^t - s_i^{fixed}\\right|$')
    plt.grid(True)
    plt.xlabel('t')
    outside_right_legend(width_squeeze=0.65)
    plt.show()


def create_mnist_figure():

    ex_continuous = X_1hid_vanilla
    ex_binary = X_osa_longer_gp_params

    # ex_continuous = experiment_mnist_eqprop_torch.get_variant('vanilla').get_variant('one_hid_swapless')
    # ex_binary = experiment_mnist_eqprop_torch.get_variant('quantized').get_variant('one_hid_swapless').get_variant(epsilons='0.843/t**0.0923', lambdas='0.832/t**0.584', quantizer='sigma_delta')
    # ex_binary = experiment_mnist_eqprop_torch.get_variant('quantized').get_variant('one_hid').get_variant(epsilons='0.843/t**0.0923', lambdas='0.832/t**0.584', quantizer='sigma_delta')

    rec_continuous = ex_continuous.get_latest_record(if_none='run')
    rec_binary = ex_binary.get_latest_record(if_none='run')

    print(f"Continuous: {report_score_from_result(rec_continuous.get_result())}")
    print(f"Binary: {report_score_from_result(rec_binary.get_result())}")

    plt.figure(figsize=(4.5, 3))
    set_figure_border_size(bottom=0.15, left=0.15)
    ex_binary.compare([rec_continuous, rec_binary], show_now=False)

    plt.legend(['Continuous Eq-Prop: Train', 'Continuous Eq-Prop: Test', 'Binary Eq-Prop: Train', 'Binary Eq-Prop: Test'])
    plt.show()


def create_gp_convergence_search_figure():

    searches = [
        # ('$\epsilon = 1/\sqrt{t}$', 'demo_quantized_convergence_perturbed.nonadaptive.epsilons=1_SLASH_sqrt(t)'),
        ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.nonadaptive.poly_schedule.parameter_search'),
        ('$\epsilon$ = OSA($\\bar\\nu$)', 'demo_quantized_convergence_perturbed.adaptive.optimal.parameter_search'),
        ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', 'demo_quantized_convergence_perturbed.adaptive.optimal.parameter_search'),
        ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.adaptive.optimal.OSA-lambda_sched.parameter_search_predictive'),
    ]

    relabels = {'eps_init': '$\epsilon_0$', 'eps_exp':' $\eta_\epsilon$', 'lambda_init': '$\lambda_0$', 'lambda_exp':' $\eta_\lambda$',
                'error_stepsize_target': '$\\bar\\nu$', 'lambdas': '$\lambda$'}
    plt.figure(figsize=(7, 7))
    set_figure_border_size(hspace=0.25, border=0.05, bottom=.1, right=0.1)

    for search_name, search_exp_id in searches:
        add_subplot()
        exp = load_experiment(search_exp_id)
        rec = exp.get_latest_record()
        plot_hyperparameter_search(rec, relabel=relabels, assert_all_relabels_used=False, score_name='Error')
        plt.title(search_name)
    plt.show()


def create_gp_mnist_search_figure():

    # TODO: Replace the first one with the search with LONGER when it's done.
    searches = [
        ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', 'experiment_mnist_eqprop_torch.1_hid.quantized.poly_schedule.epoch_checkpoint_period=None,n_epochs=1,quantizer=sigma_delta.parameter_search'),
        # ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', X_polyscheduled_longer)
        # ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', 'experiment_mnist_eqprop_torch.1_hid.adaptive_quantized.optimal.n_negative_steps=100,n_positive_steps=50.epoch_checkpoint_period=None,n_epochs=1.parameter_search')
        ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', X_osa_longer_1hid_psearch)
    ]

    relabels = {'eps_init': '$\epsilon_0$', 'eps_exp':' $\eta_\epsilon$', 'lambda_init': '$\lambda_0$', 'lambda_exp':' $\eta_\lambda$',
                'error_stepsize_target': '$\\bar\\nu$', 'lambdas': '$\lambda$'}

    plt.figure(figsize=(7, 4))
    set_figure_border_size(hspace=0.25, border=0.05, bottom=.1, right=0.1)
    for search_name, search_exp_id in searches:
        add_subplot()
        exp = load_experiment(search_exp_id) if isinstance(search_exp_id, str) else search_exp_id  # type: Experiment
        rec = exp.get_latest_record()
        plot_hyperparameter_search(rec, relabel=relabels, assert_all_relabels_used=False, score_name='Val. Error\n after 1 epoch')
        plt.title(search_name)

    plt.show()


if __name__ == '__main__':
    # create_neuron_figure()
    # create_convergence_var_figure()
    create_mnist_figure()
    # create_gp_convergence_search_figure()
    # create_gp_mnist_search_figure()
    # {1: create_neuron_figure, 2: create_convergence_var_figure, 3: create_mnist_figure}[int(input('Which figure would you like to create?  (1-3) >> '))]()
