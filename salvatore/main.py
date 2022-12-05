# Main (used for running tests)
from salvatore.utils import Logger
from salvatore.criterions import *
from salvatore.contours import test_abs_error, test_table_target_points_nn, test_double_nn, test_lines_nn


def tests_abs_error(*test_nums: int):
    pass


def tests_table_nn(*test_nums: int):
    if 0 in test_nums:
        test_table_target_points_nn(
            dir_path='..', max_generations=10000, num_of_points=4000, save_image_gen_step=50,
            logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
            stopping_criterions={
                max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 100.},
                min_fitness_percentage_gain_stop: {'percentage': 0.0001},
                flat_percentage_fitness_stop: {'epsilon_perc': 0.001, 'gen_num': 500}
            }
        )
    if 1 in test_nums:
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


def tests_double_nn(*test_nums: int):
    if 0 in test_nums:
        test_double_nn(
            dir_path='..', max_generations=4000, num_of_points=3000, device='gpu',
            target_candidate_weight=2.0, candidate_target_weight=1.0, save_image_gen_step=50,
            logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
            stopping_criterions={
                max_time_stop: {'max_time': 240. * 60.}, min_fitness_stop: {'min_fitness_value': 2000.},
                min_fitness_percentage_gain_stop: {'percentage': 0.001},
                flat_percentage_fitness_stop: {'epsilon_perc': 0.005, 'gen_num': 500}
            }
        )
    if 1 in test_nums:
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


def tests_lines_nn(*test_nums: int):
    if 0 in test_nums:
        test_lines_nn(
            dir_path='..', max_generations=10000, lineno=600, save_image_gen_step=50,
            point_adherence_coeff=1., line_adherence_coeff=5., line_l1_lambda=.0001,
            logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
            stopping_criterions={
                max_time_stop: {'max_time': 300. * 60.}, min_fitness_stop: {'min_fitness_value': 500.},
                min_fitness_percentage_gain_stop: {'percentage': 0.002},
                flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
            }
        )
    if 1 in test_nums:
        test_lines_nn(
            dir_path='..', max_generations=10000, lineno=600, save_image_gen_step=50,
            point_adherence_coeff=1., line_adherence_coeff=1., line_l1_lambda=.0001,
            logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
            stopping_criterions={
                max_time_stop: {'max_time': 300. * 60.}, min_fitness_stop: {'min_fitness_value': 500.},
                min_fitness_percentage_gain_stop: {'percentage': 0.002},
                flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
            }
        )
    if 2 in test_nums:
        test_lines_nn(
            dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000, lineno=1000,
            save_image_gen_step=50, point_adherence_coeff=1., line_adherence_coeff=5., line_l1_lambda=.0001,
            logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
            stopping_criterions={
                max_time_stop: {'max_time': 360. * 60.}, min_fitness_stop: {'min_fitness_value': 1000.},
                min_fitness_percentage_gain_stop: {'percentage': 0.002},
                flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
            }
        )
    if 3 in test_nums:
        test_lines_nn(
            dir_path='..', image_path='images/Mona_Lisa_head.png', max_generations=10000, lineno=1000,
            save_image_gen_step=50, point_adherence_coeff=1., line_adherence_coeff=1., line_l1_lambda=.0001,
            logger=Logger(dir_path='.', stats_gen_step=50, csv_gen_step=50, stats_fields=('min', 'avg')),
            stopping_criterions={
                max_time_stop: {'max_time': 360. * 60.}, min_fitness_stop: {'min_fitness_value': 1000.},
                min_fitness_percentage_gain_stop: {'percentage': 0.002},
                flat_percentage_fitness_stop: {'epsilon_perc': 0.002, 'gen_num': 500}
            }
        )


if __name__ == '__main__':
    # Uncomment the following that you want to test
    tests_abs_error()
    tests_table_nn(1)
    tests_double_nn()
    tests_lines_nn()
    exit(0)
