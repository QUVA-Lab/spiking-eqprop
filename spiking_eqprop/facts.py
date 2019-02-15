from artemis.experiments.experiment_record_view import compare_timeseries_records
from spiking_eqprop.demo_mnist_eqprop import experiment_plain_old_eqprop
from spiking_eqprop.demo_mnist_quantized_eqprop import experiment_quantized_eqprop
from matplotlib import pyplot as plt


def for_quantized_eqprop_swap_direction_has_a_problem():
    """
    Yeah, for whatever reason the direction-swap thing aint working.

    Seems to be the case for both quantized and nonquantized.
    """
    swapped_result = experiment_plain_old_eqprop.get_variant('one_hid').get_latest_record()
    unswapped_result = experiment_plain_old_eqprop.get_variant('one_hid_swapless').get_latest_record()
    compare_timeseries_records([swapped_result, unswapped_result], yfield='test_error', xfield='epoch', ax=plt.subplot(2,1,1), hang=False)

    q_swapped_result = experiment_quantized_eqprop.get_variant('one_hid').add_variant(epsilons='0.822/t**0.367', lambdas='0.17/t**0.17', quantizer='sigma_delta').get_latest_record()
    q_unswapped_result = experiment_quantized_eqprop.get_variant('one_hid_swapless').add_variant(epsilons='0.822/t**0.367', lambdas='0.17/t**0.17', quantizer='sigma_delta').get_latest_record()
    compare_timeseries_records([q_swapped_result, q_unswapped_result], yfield='test_error', xfield='epoch', ax=plt.subplot(2,1,2))


if __name__ == '__main__':
    for_quantized_eqprop_swap_direction_has_a_problem()