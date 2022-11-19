#run different quick experiments

# python3 lorenzo/genetic_from_text.py -n_gen 2000 -pop_n 50 -m_c 0.2 -m_s 2 -e True -e_n 6
# python3 lorenzo/genetic_from_text.py -n_gen 2000 -pop_n 50 -m_c 0.2 -m_s 4 -e True -e_n 6
# python3 lorenzo/genetic_from_text.py -n_gen 2000 -pop_n 50 -m_c 0.2 -m_s 8 -e True -e_n 6

# python3 lorenzo/genetic_from_text.py -n_gen 2000 -pop_n 50 -m_c 0.3 -m_s 2 -e True -e_n 6
# python3 lorenzo/genetic_from_text.py -n_gen 2000 -pop_n 50 -m_c 0.4 -m_s 2 -e True -e_n 6
# python3 lorenzo/genetic_from_text.py -n_gen 2000 -pop_n 50 -m_c 0.5 -m_s 2 -e True -e_n 6

python3 lorenzo/genetic_from_text.py -d mse  -n_gen 50 -pop_n 50 -m_c 0.1 -m_s 1 -e True -e_n 5
python3 lorenzo/genetic_from_text.py -d psnr  -n_gen 50 -pop_n 50 -m_c 0.1 -m_s 1 -e True -e_n 5

