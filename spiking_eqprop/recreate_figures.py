from matplotlib import pyplot as plt

from artemis.experiments import load_experiment
from artemis.experiments.experiment_management import get_multiple_records
from artemis.plotting.expanding_subplots import set_figure_border_size, add_subplot
from artemis.plotting.pyplot_plus import outside_right_legend
from artemis.plotting.range_plots import plot_sample_mean_and_var
from spiking_eqprop.demo_quantized_convergence_with_perturbation import demo_quantized_convergence_perturbed
from spiking_eqprop.demo_mnist_eqprop import report_score_from_result
from spiking_eqprop.demo_mnist_quantized_eqprop import experiment_mnist_eqprop_torch
from spiking_eqprop.demo_signal_figure import demo_create_signal_figure


def create_neuron_figure():

    demo_create_signal_figure()


# def create_convergence_figure():
#
#     X = demo_quantized_convergence_torch.get_variant('scheduled')
#
#     records = [
#         X.get_variant('epsilon_decay').get_variant(epsilons='1/2').get_latest_record(if_none='run', only_completed=True),
#         X.get_variant('epsilon_decay').get_variant(epsilons='1/t').get_latest_record(if_none='run', only_completed=True),
#         X.get_variant('epsilon_decay').get_variant(epsilons='1/sqrt(t)').get_latest_record(if_none='run', only_completed=True),
#         X.get_variant(epsilons ='1/sqrt(t)', quantizer ='stochastic').get_latest_record(if_none='run', only_completed=True),
#         X.get_variant('search').get_variant('epsilons=0.843_SLASH_t**0.0923,lambdas=0.832_SLASH_t**0.584').get_latest_record(if_none='run', only_completed=True),  # Best Mean Error
#         ]
#     compare_quantized_convergence_records(
#             records = records,
#             ax = add_subplot(), show_now=False, include_legend_now = True, label_reference_lines=True, legend_squeeze=0.5
#             )
#     plt.xlabel('t')
#     plt.show()


def create_convergence_perturbed_figure():

    X = demo_quantized_convergence_perturbed.get_variant('scheduled')

    for v in demo_quantized_convergence_perturbed.get_all_variants():
        print(v.name)

    # records = [
    #     ('$\epsilon = 1/\sqrt{t}$', 'demo_quantized_convergence_perturbed.test_adaptive.nonadaptive.epsilons=1_SLASH_sqrt(t)'),
    #     ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.test_adaptive.nonadaptive.poly_schedule.gp_params'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$)', 'demo_quantized_convergence_perturbed.test_adaptive.adaptive.optimal.gp_params'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', 'demo_quantized_convergence_perturbed.test_adaptive.adaptive.optimal.gp_params_predictive'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.test_adaptive.adaptive.optimal.OSA-lambda_sched.gp_params'),
    # ]
    # records = [
    #     ('$\epsilon = 1/\sqrt{t}$', 'demo_quantized_convergence_perturbed.test_adaptive.nonadaptive.epsilons=1_SLASH_sqrt(t)'),
    #     ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.test_adaptive.nonadaptive.poly_schedule.gp_params_end'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$)', 'demo_quantized_convergence_perturbed.test_adaptive.adaptive.optimal.gp_params_end'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', 'demo_quantized_convergence_perturbed.test_adaptive.adaptive.optimal.gp_params_predictive_end'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.test_adaptive.adaptive.optimal.OSA-lambda_sched.gp_params_end'),
    # ]

    for name, exp_id in records:
        record = load_experiment(exp_id).get_latest_record(only_completed=True, if_none='run')
        errors, _, _ = record.get_result()
        plt.semilogy(errors.mean(axis=1), label=name)
        all_args = record.get_args()
        perturbation_step = int(all_args['n_steps'] * all_args['change_frac'])-1
    plt.axvline(perturbation_step, color='k', linestyle = '--', linewidth=2)
    plt.ylabel('$\\frac{1}{|S|}\sum_{i \in S}\left|s_i^t - s_i^{fixed}\\right|$')
    plt.grid(True)
    # records = [
    #     X.get_variant('epsilon_decay').get_variant(epsilons='1/2').get_latest_record(if_none='run'),
    #     X.get_variant('epsilon_decay').get_variant(epsilons='1/t').get_latest_record(if_none='run'),
    #     X.get_variant('epsilon_decay').get_variant(epsilons='1/sqrt(t)').get_latest_record(if_none='run'),
    #     X.get_variant(epsilons ='1/sqrt(t)', quantizer ='stochastic').get_latest_record(if_none='run'),
    #     X.get_variant('search').get_variant('epsilons=0.843_SLASH_t**0.0923,lambdas=0.832_SLASH_t**0.584').get_latest_record(if_none='run'),  # Best Mean Error
    #     ]
    # compare_quantized_convergence_records(
    #         records = records,
    #         ax = add_subplot(), show_now=False, include_legend_now = True, label_reference_lines=True, legend_squeeze=0.5
    #         )
    plt.xlabel('t')
    outside_right_legend(width_squeeze=0.65)
    plt.show()



def create_convergence_var_figure(n=20):

    X = demo_quantized_convergence_perturbed.get_variant('scheduled')

    for v in demo_quantized_convergence_perturbed.get_all_variants():
        print(v.name)

    # experiments = [
    #     ('$\epsilon = 1/\sqrt{t}$', 'demo_quantized_convergence_perturbed.nonadaptive.epsilons=1_SLASH_sqrt(t)'),
    #     ('$\epsilon = \epsilon_0/t^{\eta_\epsilon}, \lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.nonadaptive.poly_schedule.gp_params'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$)', 'demo_quantized_convergence_perturbed.adaptive.optimal.gp_params'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda$', 'demo_quantized_convergence_perturbed.adaptive.optimal.gp_params_predictive'),
    #     ('$\epsilon$ = OSA($\\bar\\nu$), $\lambda = \lambda_0/t^{\eta_\lambda}$', 'demo_quantized_convergence_perturbed.adaptive.optimal.OSA-lambda_sched.gp_params'),
    # ]

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

        # record = load_experiment(exp_id).get_latest_record(only_completed=True, if_none='run')
        # errors, _, _ = record.get_result()
        # plt.semilogy(errors.mean(axis=1), label=name)

        plot_sample_mean_and_var(err, label=name, fill_alpha=0.15)

        all_args = exp.get_args()
        perturbation_step = int(all_args['n_steps'] * all_args['change_frac'])-1
    plt.axvline(perturbation_step, color='k', linestyle = '--', linewidth=2)
    plt.ylabel('$\\frac{1}{|S|}\sum_{i \in S}\left|s_i^t - s_i^{fixed}\\right|$')
    plt.grid(True)
    # records = [
    #     X.get_variant('epsilon_decay').get_variant(epsilons='1/2').get_latest_record(if_none='run'),
    #     X.get_variant('epsilon_decay').get_variant(epsilons='1/t').get_latest_record(if_none='run'),
    #     X.get_variant('epsilon_decay').get_variant(epsilons='1/sqrt(t)').get_latest_record(if_none='run'),
    #     X.get_variant(epsilons ='1/sqrt(t)', quantizer ='stochastic').get_latest_record(if_none='run'),
    #     X.get_variant('search').get_variant('epsilons=0.843_SLASH_t**0.0923,lambdas=0.832_SLASH_t**0.584').get_latest_record(if_none='run'),  # Best Mean Error
    #     ]
    # compare_quantized_convergence_records(
    #         records = records,
    #         ax = add_subplot(), show_now=False, include_legend_now = True, label_reference_lines=True, legend_squeeze=0.5
    #         )
    plt.xlabel('t')
    outside_right_legend(width_squeeze=0.65)
    plt.show()



def create_mnist_figure():

    ex_continuous = experiment_mnist_eqprop_torch.get_variant('vanilla').get_variant('one_hid_swapless')
    ex_binary = experiment_mnist_eqprop_torch.get_variant('quantized').get_variant('one_hid_swapless').get_variant(epsilons='0.843/t**0.0923', lambdas='0.832/t**0.584', quantizer='sigma_delta')
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


if __name__ == '__main__':
    # create_convergence_perturbed_figure()
    create_convergence_var_figure()
    # {1: create_neuron_figure, 2: create_convergence_figure, 3: create_mnist_figure}[int(input('Which figure would you like to create?  (1-3) >> '))]()
