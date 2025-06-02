import os, sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(f'{current_dir}/../../src')

from state_assignment import compute_energy_for_given_state
import my_constants as mc
import pickle


def main():
    chV_level = 'level_1_chV_1_structureNum_0'
    con = 'GLN'
    with open('resState.pkl', 'rb') as f:
        resState = pickle.load(f)
    if os.path.isdir('outputs_test') is False:
        os.makedirs('outputs_test')
    mc.protID = 'test'
    compute_energy_for_given_state(resState, chV_level, con)



if __name__ == "__main__":
    main()
