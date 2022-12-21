# Main (used for running tests)
from salvatore.utils import Logger
from salvatore.criterions import *
from salvatore.contours import test_table_target_points_nn, test_lines_nn, \
    test_table_target_points_overlap_penalty
# from salvatore.contours.ann_extra import *


def test_table_eiffel_tower(use_cython=True):
    test_table_target_points_nn(
        dir_path='..', max_generations=12000, num_of_points=4000, save_image_gen_step=100, use_gpu=True,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 72. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }, use_cython=use_cython,
    )


def tests_table_mona_lisa(use_cython=True):
    test_table_target_points_nn(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000,
        num_of_points=6000, save_image_gen_step=100, canny_low=150, canny_high=200, use_gpu=True,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 72. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0005},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.005, 'gen_num': 500}
        }, use_cython=use_cython,
    )


def test_lines_nn_eiffel_tower(use_cython=True):
    test_lines_nn(
        dir_path='..', max_generations=10000, lineno=600, save_image_gen_step=100,
        point_adherence_coeff=1.0, line_adherence_coeff=5.0, line_l1_lambda=0.0,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }, use_cython=use_cython,
    )


def test_lines_nn_mona_lisa(use_cython=True):
    test_lines_nn(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000, lineno=1000,
        save_image_gen_step=100, point_adherence_coeff=1.0, line_adherence_coeff=5.0, line_l1_lambda=0.0,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 1000.},
            min_fitness_percentage_gain_stop: {'percentage': 0.002},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
        }, use_cython=use_cython,
    )


def test_table_op_eiffel_tower(use_cython=True):
    test_table_target_points_overlap_penalty(
        dir_path='..', max_generations=10000, num_of_points=4000, save_image_gen_step=100,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }, use_cython=use_cython,
    )


def tests_table_op_mona_lisa(use_cython=True):
    test_table_target_points_overlap_penalty(
        dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000,
        num_of_points=6000, save_image_gen_step=100, canny_low=150, canny_high=200,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }, use_cython=use_cython,
    )


"""
def test_table_ann_eiffel_tower():
    test_table_target_points_ann(
        dir_path='..', max_generations=50_000, num_of_points=4000, save_image_gen_step=100,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 120. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }
    )


def test_table_ann_salvatore():
    test_table_target_points_ann(
        dir_path='..', image_path='images/salvatore.png', max_generations=50_000, num_of_points=4000,
        save_image_gen_step=100, canny_low=50, canny_high=100,
        logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 120. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }
    )
"""


def test_table_singapore(resolution: str = '400x300', use_cython=True):
    test_table_target_points_nn(
        dir_path='..', image_path=f'images/Singapore_skyline_{resolution}.jpg', max_generations=50_000,
        num_of_points=10_000, save_image_gen_step=100, canny_low=100, canny_high=200, use_gpu=True,
        logger=Logger(dir_path='.', stats_gen_step=100, csv_gen_step=100, stats_fields=('min', 'avg')),
        stopping_criterions={
            max_time_stop: {'max_time': 180. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
            min_fitness_percentage_gain_stop: {'percentage': 0.0001},
            flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
        }, use_cython=use_cython,
    )


if __name__ == '__main__':
    # Uncomment the following that you want to test
    # test_table_singapore(use_cython=True)
    # test_table_eiffel_tower(use_cython=True)
    test_lines_nn_eiffel_tower(use_cython=True)
    """
    test_table_op_eiffel_tower(use_cython=True)
    test_table_eiffel_tower(use_cython=True)
    tests_table_mona_lisa(use_cython=True)
    test_double_nn_eiffel_tower(use_cython=True)
    test_double_nn_mona_lisa(use_cython=True)
    test_lines_nn_eiffel_tower(use_cython=True)
    test_lines_nn_mona_lisa(use_cython=True)
    """
    exit(0)
