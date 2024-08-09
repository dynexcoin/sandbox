# Import Supersudoku Libary to further import all packages as required
from supersudoku import *

read_solve_write_display_sudoku(input_file_name='supersudoku/sudokuDim4NothingFixed.json', solver='dynex', 
    config=get_config(dim=4), wrap_in_html=True, popup_solution=False)