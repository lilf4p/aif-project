# Main (used for running tests)
from salvatore.utils import Logger
from salvatore.criterions import *
from salvatore.contours import test_table_target_points_nn, test_double_nn, test_lines_nn, \
    test_table_target_points_overlap_penalty


def test_table_eiffel_tower():
    test_table_target_points_nn(
        dir_path='..', max_generations=10000, num_of_points=4000, save_image_gen_step=50,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 0.2 * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }
    )


def tests_table_mona_lisa():
    test_table_target_points_nn(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000,
        num_of_points=6000, save_image_gen_step=50, canny_low=150, canny_high=200,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 420. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0005},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.005, 'gen_num': 500}
        }
    )


def test_double_nn_eiffel_tower():
    test_double_nn(
        dir_path='..', max_generations=2000, num_of_points=3000, device='gpu',
        target_candidate_weight=2.0, candidate_target_weight=1.0, save_image_gen_step=50,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 0.2 * 60.}, min_fitness_stop: {'min_fitness_value': 2000.},
            min_fitness_percentage_gain_stop: {'percentage': 0.001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.005, 'gen_num': 500}
        }
    )


def test_double_nn_mona_lisa():
    test_double_nn(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=4000,
        num_of_points=3000, device='gpu', save_image_gen_step=50,
        target_candidate_weight=2.0, candidate_target_weight=1.0,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 2000.},
            min_fitness_percentage_gain_stop: {'percentage': 0.001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.005, 'gen_num': 500}
        }
    )


def test_lines_nn_eiffel_tower():
    test_lines_nn(
        dir_path='..', max_generations=10000, lineno=600, save_image_gen_step=50,
        point_adherence_coeff=1., line_adherence_coeff=5., line_l1_lambda=0.,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 0.2 * 60.}, min_fitness_stop: {'min_fitness_value': 500.},
            min_fitness_percentage_gain_stop: {'percentage': 0.002},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
        }
    )


def test_lines_nn_mona_lisa():
    test_lines_nn(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000, lineno=1000,
        save_image_gen_step=50, point_adherence_coeff=1., line_adherence_coeff=5., line_l1_lambda=0.,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 360. * 60.}, min_fitness_stop: {'min_fitness_value': 1000.},
            min_fitness_percentage_gain_stop: {'percentage': 0.002},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
        }
    )


def test_table_op_eiffel_tower():
    test_table_target_points_overlap_penalty(
        dir_path='..', max_generations=10000, num_of_points=4000, save_image_gen_step=50,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 30. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }
    )


def tests_table_op_mona_lisa():
    test_table_target_points_overlap_penalty(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=500,
        num_of_points=6000, save_image_gen_step=50, canny_low=150, canny_high=200,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 30. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0005},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.005, 'gen_num': 500}
        }
    )


if __name__ == '__main__':
    # Uncomment the following that you want to test
    # test_table_eiffel_tower()
    # test_double_nn_eiffel_tower()
    test_lines_nn_eiffel_tower()
    """
    tests_table_op_mona_lisa()
    test_table_op_eiffel_tower()
    test_table_eiffel_tower()
    tests_table_mona_lisa()
    test_double_nn_eiffel_tower()
    test_double_nn_mona_lisa()
    test_lines_nn_eiffel_tower()
    test_lines_nn_mona_lisa()
    """
    exit(0)
