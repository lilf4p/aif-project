# Main (used for running tests)
from salvatore.utils import Logger
from salvatore.contours import test_abs_error, test_table_target_points_nn, test_target_points_nn, \
    test_double_nn, test_lines_nn, test_distance_matrix


if __name__ == '__main__':
    # Uncomment the following that you want to test
    """
    test_abs_error(dir_path='..', max_generations=10)
    test_table_target_points_nn(dir_path='..', max_generations=10000, gen_step=25)
    test_target_points_nn(dir_path='..', max_generations=10)
    test_double_nn(dir_path='..', max_generations=10)
    test_table_target_points_nn(dir_path='..', image_path='images/Mona_lisa_head.png',
                                max_generations=1000, gen_step=50, num_of_points=5000)
    """
    test_lines_nn(dir_path='..', max_generations=13, lineno=500, save_image_gen_step=5, stats_gen_step=5,
                  line_l1_lambda=0.01, logger=Logger(dir_path='.', stats_gen_step=5,
                                                     csv_gen_step=2, stats_fields=('min', 'avg')
                                                     )
                  )
    # test_distance_matrix(dir_path='..', max_generations=2000, num_of_points=1200, gen_step=50)
    exit(0)
