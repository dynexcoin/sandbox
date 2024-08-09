# Super Sudoku Solver Libary
# Original Codebase from - https://github.com/PeterSels/OnComputation
# Adapted to run on Dynex n.quantum by Y3TI & Sam Rahmeh
from IPython.core.display import HTML, display
import sys
import os
import json
import math
import numpy as np
from collections import defaultdict
import itertools
import dimod
import dynex
import datetime
import time
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite, LeapHybridSampler
from minorminer import find_embedding
from gurobipy import *

def read_sudoku_from_json_file_and_check(file_name):
    verbose = 0  # set to 1 to see more output
    sudoku_json = json.load(open(file_name))
    # do some basic checks to see we have all the information needed and none other
    errors = ''
    # read dim(ension) field
    dim = int(sudoku_json["dim"])
    if verbose > 0:
        print("dim = {:d}".format(dim))
    max_nr = pow(dim, 2)
    if verbose > 0:
        print("max_nr = {:d}".format(max_nr) if (verbose > 0) else '')
    nrs = list(range(1, max_nr + 1))
    nrs_str = '[' + ','.join([str(nr) for nr in nrs]) + ']'
    if verbose > 0:
        print(nrs_str if (verbose > 0) else '')

    # read fixed part
    fixed = sudoku_json["fixed"]
    for row in fixed:
        r = int(row)
        if r not in nrs:
            errors += \
                'row index number should be in {:s} but is {:d}.\n'. \
                    format(nrs_str, r)
        for col in fixed[row]:
            c = int(col)
            if c not in nrs:
                errors += 'column index number should be in {:s} but is {:d}.\n'. \
                    format(nrs_str, c)
            num = fixed[row][col]
            n = int(num)
            if n not in nrs:
                errors += 'square[{:d}][{:d}] number should be in {:s} but is {:d}.\n'. \
                    format(r, c, nrs_str, n)

    print('I have read a ' + ('faulty' if (errors != '') else 'valid') + \
          ' MetaSudoku problem description of dimension {:d}.'.format(dim))
    if len(errors) > 0:
        print('\nerrors:' + errors + '\n')

    n_bin_vars = dim ** 5
    print('This will require {:d}^5 = {:d} binary variables in our model we are setting up now.'.\
          format(dim, n_bin_vars))
    return dim, fixed

def rcn_to_qi(dim: int, r: int, c:int, n:int):  # dim=3 for a standard 3^2 * 3^2 sudoku
    # This function takes the row, column and digit index of the binary (qubit)
    # variables b_{r,c,n} and maps them to the 2D QUBO matrix.
    dim_sq = dim ** 2  # is 9 for a standard 9 x 9 sudoku
    assert dim >= 0
    assert r >= 0
    assert c >= 0
    assert n >= 0
    assert r <= dim_sq-1
    assert c <= dim_sq-1
    assert n <= dim_sq-1
    do_val1 = False
    if do_val1:
        val1 = (dim_sq ** 2 * r) + (dim_sq * c) + n
        print('val1 = ', val1)
    val2 = dim_sq * ((dim_sq * r) + c) + n   # Do it the Horner way
    if do_val1:
        print('val2 = ', val2)
        assert val1 == val2
    assert val2 <= dim_sq ** 3 - 1
    return val2

def test_rcn_to_qi():
    assert rcn_to_qi(3, 0, 0, 0) == 0
    assert rcn_to_qi(3, 0, 0, 1) == 1
    assert rcn_to_qi(3, 1, 1, 1) == (9 ** 2 * 1) + (9 * 1) + 1
    print('test_rcn_to_qi went ok.')

def print_qubo(Q):  # This function takes a forbiddingly huge amount of time for n>=6. Only call it for smaller n.
    i, j = zip(*Q.keys())
    np_matrix = np.zeros((max(i) + 1, max(j) + 1), np.int32)
    np.add.at(np_matrix, tuple((i, j)), tuple(Q.values()))
    np.set_printoptions(threshold=np.inf)
    print('QUBO Q=\n', np_matrix)

# tested with [Python 3.7.6 & 3.8] & Gurobi 9 for solver == 'gurobi'
def solve_sudoku(dim, fixed, solver, config, print_num_vals=False):
    model_build_start = time.perf_counter()
    verbose = 1
    n_rows = n_cols = n_nums = dim * dim
    n_subs = dim
    num_reads = config['num_reads']   
    rows = cols = nums = list(range(n_rows))
    subs = list(range(n_subs))
    if verbose > 0:
        print('subs = ', subs)

    if verbose > 0:
        print('rows = ', rows)
        print('cols = ', cols)
        print('nums = ', nums)
        print('subs = ', subs)

    Q = defaultdict(int)
    # define the basic Constraints
    if config['num'] > 0:
        lagrange_weight = 2
        for r in rows:
            for c in cols:
                for n0 in nums:
                    i0 = rcn_to_qi(dim, r, c, n0)
                    Q[(i0, i0)] += -1 * config['num']
                    for n1 in nums:
                        i1 = rcn_to_qi(dim, r, c, n1)
                        if i1 > i0:
                            Q[(i0, i1)] += 2 * config['num']                                

    if config['row'] > 0:
        for r in rows:
            for n in nums:
                for c0 in cols:
                    i0 = rcn_to_qi(dim, r, c0, n)
                    Q[(i0, i0)] += -1 * config['row']
                    for c1 in cols:
                        i1 = rcn_to_qi(dim, r, c1, n)
                        if i1 > i0:
                            Q[(i0, i1)] += 2 * config['row']                                

    if config['col'] > 0:
        for c in cols:
            for n in nums:
                for r0 in rows:
                    i0 = rcn_to_qi(dim, r0, c, n)
                    Q[(i0, i0)] += -1 * config['col']
                    for r1 in rows:
                        i1 = rcn_to_qi(dim, r1, c, n)
                        if i1 > i0:
                            Q[(i0, i1)] += 2 * config['col']                                

    if config['sub'] > 0:
        combos = list(itertools.product(*[subs, subs]))
        if verbose > 0:
            print('combos = ', combos)

        for r in subs:
            for c in subs:
                for n in nums:
                    for r0_, c0_ in combos:
                        r0__, c0__ = r * dim + r0_, c * dim + c0_
                        i0 = rcn_to_qi(dim, r0__, c0__, n)
                        Q[(i0, i0)] += -1 * config['sub']
                        for r1_, c1_ in combos:
                            r1__, c1__ = r * dim + r1_, c * dim + c1_
                            i1 = rcn_to_qi(dim, r1__, c1__, n)
                            if i1 > i0:
                                Q[(i0, i1)] += 2 * config['sub']                                    

    if config['fix'] > 0:
        # initial squares, fixed
        for r_str in fixed:
            r = int(r_str) - 1  # externally, we have a 1-offset for sudoku digits, internally we have a 0-offset 
            for c_str in fixed[r_str]:
                c = int(c_str) - 1 # externally, we have a 1-offset for sudoku digits, internally we have a 0-offset
                f = int(fixed[r_str][c_str])
                #print(f'*** f={f}')
                for n0 in [f-1]:  # externally, we have a 1-offset for sudoku digits, internally we have a 0-offset
                    #print(f'*** n0={n0}')
                    i0 = rcn_to_qi(dim, r, c, n0)
                    #print(f'*** i0={i0}')
                    Q[(i0, i0)] += -1 * config['fix']  # MIP bi==1 becomes (bi-1)^2 becomes bi^2 -2bi + 1^2
                    # which is -bi + 1 for binary bi, so we get term -1 only added to the QUBO diagonal                

    define_integer_vars_in_model_io_in_postprocessing = True
    model_build_end = time.perf_counter()
    print(f">>> model_build process took {model_build_end - model_build_start:0.4f} seconds.")

    print('dim = {:d}'.format(dim))
    if dim <= 2:
        print_qubo(Q)
    else:
        print('  Not printing nor checking matrix, since dim >= 3 is too large to be useful.')

    assert config['method'] == 'hybrid'
    bqm = qubo_to_bqm(Q, dim)
    find_embedding_end = time.perf_counter()
    print(f">>> find_embedding process took {find_embedding_end - model_build_end:0.4f} seconds.")
    
    sample_set = run_on_hybrid("dynex", bqm, dim)
    print('  sample_set returned.')
    print('# samples received = ', len(sample_set))
    print(sample_set)        
     
    model_solve_end = time.perf_counter()
    print(f">>> solve process took {model_solve_end - find_embedding_end:0.4f} seconds.")
    
    # retrieve solution
    print('  calc_numvals_from_dynex_samples...')
    num_vals = calc_numvals_from_dynex_samples(sample_set, dim)
    print('  calc_numvals_from_dynex_samples done.')

    if print_num_vals:
        print('num_vals = ', num_vals)

    get_num_vals_end = time.perf_counter()
    print(f">>> get_num_vals process took {get_num_vals_end - model_solve_end:0.4f} seconds.")
        
    print('  verify_solution check...')
    sudoku_ok, presets_ok = verify_solution(num_vals, dim, config, fixed)
    print('  verify_solution check done.')
    if sudoku_ok:
        print('Sudoku rules are respected in solution. :)')
    else:
        print('ERROR: Sudoku rules are NOT respected in solution. :(')
        #print('Only the first error is reported here. There could be more. :(')
        print('We still want to continue here to produce a json and a html file of the sudoku \'solution\'.')
    if presets_ok:
        print('Sudoku preset square values are respected in solution. :)')
    else:
        print('ERROR: Sudoku preset square values are NOT respected in solution. :(')

    verify_num_vals_end = time.perf_counter()
    print(f"verify_num_vals process took {verify_num_vals_end - get_num_vals_end:0.4f} seconds.")
    return rows, cols, subs, num_vals

def verify_solution(num_vals, dim, config, fixed):
    """ Verify that the found solution satisfies the Sudoku constraints. """    
    sudoku_ok, presets_ok = True, True
    
    verbose = 0  # set to 1 to see more output
    dim_sq = dim ** 2
    print('    * dim = ', dim)
    print('    * dim_sq = ', dim_sq)
    rows = cols = list(range(dim_sq))  #int(math.sqrt(len(num_vals)))  # Number of rows/columns
    subs = list(range(dim))
    #assert math.sqrt(rows).is_integer()
    unique_digits = set(range(1, dim_sq + 1))  # Digits in a solution
    if verbose > 0:
        print('    * unique digits:', unique_digits)

    if config['row'] > 0:
        # Verifying rows
        print('    * rows = ', rows)
        for r in rows:
            s = set()
            for c in cols:
                s.add(num_vals[(r, c)])
            if s != unique_digits:
                print('    ERROR: Only {:d} (i.o. {:d}) different values in row {:d}'.format(len(s), dim_sq, r + 1))
                print('      Set = ', s)
                missing_digits_set = unique_digits.difference(s)
                print('      Missing digits set = ', missing_digits_set, '\n')
                sudoku_ok = False

    if config['col'] > 0:
        # Verifying cols
        print('    * cols = ', cols)
        for c in cols:
            s = set()
            for r in rows:
                s.add(num_vals[r, c])
            if s != unique_digits:
                print('    ERROR: Only {:d} (i.o. {:d}) different values in column {:d}'.format(len(s), dim_sq, c + 1))
                print('      Set = ', s)
                missing_digits_set = unique_digits.difference(s)
                print('      Missing digits set = ', missing_digits_set, '\n')
                sudoku_ok = False

    if config['num'] > 0:
        pass  # nothing to do here.

    if config['sub'] > 0:
        # Verifying sub-squares
        subsquare_coords = [(i, j) for i in subs for j in subs]
        print('    * subs = ', subs)
        for r_ in subs:
            for c_ in subs:
                s = [num_vals[i + r_ * dim, j + c_ * dim] for i, j in subsquare_coords]
                set_s = set(s)
                if set_s != unique_digits:
                    print("    ERROR: Only {:d} (i.o. {:d}) different values in sub-square (:d, :d): ".\
                          format(len(set_s), dim_sq, r_ + 1, c_ + 1))
                    print('      Set = ', set_s)
                    missing_digits_set = unique_digits.difference(set_s)
                    print('      Missing digits set = ', missing_digits_set, '\n')
                    sudoku_ok = False
                
    if config['fix'] > 0:
        n_presets_checked = 0
        print('    * preset squares:')  # equavalently called: 'fixed squares'
        for r_str in fixed:
            r = int(r_str) - 1  # externally, we have a 1-offset for sudoku digits, internally we have a 0-offset 
            for c_str in fixed[r_str]:
                c = int(c_str) - 1 # externally, we have a 1-offset for sudoku digits, internally we have a 0-offset
                f = int(fixed[r_str][c_str])
                n_presets_checked += 1
                sol_f = num_vals[r, c]
                if sol_f != f:
                    print(f'    ERROR: (row, col) = ({r+1}, {c+1}) has value {sol_f} in solution, but value {f} was preset.')        
                    presets_ok = False
        print(f'    INFO: {n_presets_checked} preset squares checked.')
                    
    return sudoku_ok, presets_ok

def calc_numvals_from_dynex_samples(sample_set, dim):
    num_vals = {}
    rows = cols = nums = range(dim * dim)
    print('calc_numvals_from_dynex_samples:')
    print('rows =', rows)
    n_1_qubits = 0
    n_0_qubits = 0

    n_samples = len(sample_set)
    print('    # samples in sample_set = {:d}'.format(n_samples))
    if n_samples < 1:
        print('The solver did not return a sample, so we have to quit.')
        exit(0)

    for sample in sample_set:
        for r in rows:
            for c in cols:
                rc_total_bins_on_1 = 0
                num_1_set = set()
                for n in nums:
                    i = rcn_to_qi(dim, r, c, n)
                    qubo_i_val = sample[i]
                    if qubo_i_val == 1:
                        num_vals[(r, c)] = n+1
                        n_1_qubits += 1
                        rc_total_bins_on_1 += 1
                        num_1_set.add(n)
                    else:
                        assert qubo_i_val == 0
                        n_0_qubits += 1
                    #else:
                    #    num_vals[(r, c)] = 0
                if rc_total_bins_on_1 != 1:
                    msg = '  ERROR: In result returned from solver, sudoku position (r,c) =\n'
                    msg += '   ({:d},{:d}) has {:d} binary vars on 1 i.o. just one of them.\n'.\
                        format(r, c, rc_total_bins_on_1)
                    msg += '  The set of numbers for which we have a 1 is {}\n'.format(num_1_set)
                    msg += '  This is not a valid solution. But continuing anyway...' #' Quitting.'
                    print(msg)
                    num_vals[(r, c)] = 0  # to avoid missing (r,c) keys in mum_vals later.
        break  # just look at top sample in samples list since that's the one with the lowest energy
        
    print('n_1_qubits = ', n_1_qubits)
    n_1_qubits_should_be = dim ** 4
    if n_1_qubits != n_1_qubits_should_be:
        print(f'ERROR: The number of qubits on 1 ({n_1_qubits}) is not what is should be ({n_1_qubits_should_be})')
    print('n_0_qubits = ', n_0_qubits)
    n_total_qubits = n_0_qubits + n_1_qubits
    n_total_qubits_should_be = dim ** 6
    if n_total_qubits != n_total_qubits_should_be:
        print(f'ERROR: The number of total qubits ({n_total_qubits}) is not what is should be (dim^6 = {n_total_qubits_should_be})')
    return num_vals

# To send a QUBO matrix to a hybrid d-wave system i.o. to q QPU, we need to convert the Q(UBO) matrix into a
# data structure of type 'AdjVectorBQM'. This function converts Q to that AdjVectorBQM structure.
def qubo_to_bqm(Q, dim):
    linear_dict = {}
    quadratic_dict = {}
    for (i, j) in Q.keys():
        if i == j:
            linear_dict[i] = Q[(i, j)]
        else:
            quadratic_dict[(i, j)] = Q[(i, j)]

    bqm = dimod.AdjVectorBQM(linear_dict, quadratic_dict, dimod.Vartype.BINARY)
    if dim <= 2:
        print('  bqm =\n', bqm)
    else:
        print('  Not printing nor bqm matrix, since dim >= 3 is too large for being useful,')
        print('    and at dim >= 6, it really takes a huge amount of time.')
    return bqm

# This function sends the (to type AdjVectorBQM converted) QUBO matrix to the d-wave hybrid samples.
# The hybrid sampler only returns one sample in the sample_set, but the format returned is the same
# as the one returned by the function run_on_qpu above.
def run_on_hybrid(solver, bqm, dim):
    n_qubits = dim ** 6
    print('n_qubits = {:d}'.format(n_qubits)) 
    time_limit = 1000  # default
    anneal_limit = 1000
    if dim == 4:
        time_limit = 2900 
        anneal_limit = 4700        
    
    print('n_quantum_reads = ', time_limit) 
    print('n_quantum_anneal = ', anneal_limit) 
    if dim >= 4:
        print('n_qubo_generate - this may take some time. go grab a cup of tea') 

    # Sample on Dynex n.quantum
    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, mainnet=False, description='Quantum Super Sudoku', bnb=False)
    sampleset = sampler.sample(num_reads=time_limit, annealing_time=anneal_limit, debugging=False, alpha=10, beta=0.1)
    print('  Sampling Dynex...')
    return sampleset

def solution_to_html_string(
        rows, cols, subs,
        fixed, num_vals, caption,
        wrap_in_html_object,
        print_html_string=False,
        sudoku_table_style="table { border-collapse: collapse; " + \
                           "font-family: Calibri, sans-serif; } " + \
                           "colgroup, tbody { border: solid thin; } td { td border: solid thin; " \
                           "height: 1.4em; width: 1.4em; text-align: center; padding: 0; }\n"
):
    table = '  <table>\n'
    table += '    <caption>\n'
    table += '  {:s}\n    '.format(caption)
    table += '    </caption>\n'

    N = len(subs)
    for s1 in subs:
        table += '    <colgroup>\n'
        for s2 in subs:
            table += '        <col>\n'
        table += '    </colgroup>\n'

    for r in rows:
        if (r % N) == 0:
            table += '    <tbody>\n'
        table += '      <tr>\n'
        for c in cols:
            pre = '<td style="color:black;">'
            if str(r + 1) in fixed:
                if str(c + 1) in fixed[str(r + 1)]:
                    pre = '<td style="color:red;">'
            num_str = '_'
            if (r, c) in num_vals.keys():
                num_str = '{:d} '.format(int(num_vals[(r, c)]))
            table += '        ' + pre + num_str + '</td>\n'
        if (r % N) == N - 1:
            table += '    </tbody>\n'

    table += '  </table>'

    s = ''
    s += '<html>\n'

    s += '<head>\n'
    s += '  <style>\n'
    s += '    ' + sudoku_table_style
    s += '  </style>\n'
    s += '</head>\n'

    s += '<body>\n'
    s += '  ' + table + '\n'
    s += 'Y3TI is a god\n'
    s += '</body>\n'

    s += '</html>'

    if print_html_string:
        print(s)
    return s

def write_sudoku_solution_to_json_file(dim, fixed, num_vals, output_file_name):
    d = {}
    d["dim"] = dim
    d["fixed"] = fixed  # fixed stores keys in row then col and both
    # in string form already, since we read it from json input
    d["solved"] = {}
    for (row, col) in num_vals:  # num_vals stores row, col keys
        # as an integer pair
        # print(row, col)
        row_str = str(row + 1)
        col_str = str(col + 1)
        if row_str in fixed and col_str in fixed[row_str]:
            # it's part of the fixed squares and will be written
            # out via d["fixed"]
            pass
        else:
            if not (row_str in d["solved"]):
                d["solved"][row_str] = {}
            d["solved"][row_str][col_str] = int(num_vals[(row, col)])
    with open(output_file_name, 'w') as outfile:
        json.dump(d, outfile, indent=2)

def read_solve_write_display_sudoku(input_file_name, solver, config, wrap_in_html, popup_solution):
    assert solver in ['dynex']
    dim, fixed = read_sudoku_from_json_file_and_check(input_file_name)

    rows, cols, subs, num_vals = solve_sudoku(dim, fixed, solver, config)
    output_file_name = input_file_name.replace('.json', '_solved.json')
    write_sudoku_solution_to_json_file(dim, fixed, num_vals, output_file_name)
    
    html_table = solution_to_html_string(
        rows, cols, subs, fixed, num_vals,
        caption='{:d} x {:d} x {:d} x {:d} Sudoku'.format(dim, dim, dim, dim),
        wrap_in_html_object=wrap_in_html,
        print_html_string=False
    )

    html_output_file_name = input_file_name.replace('.json', '_solved.html')
    f = open(html_output_file_name, "w")
    f.write(html_table)
    print(f'The file {html_output_file_name} is written.')
    f.close()

    if popup_solution:
        os.system('open ' + html_output_file_name)
    if wrap_in_html:
        return HTML(html_table)    
    else:
        return html_table  

def get_config(dim):
    method = 'hybrid'  # the below num_reads fields is NOT used for hybrid
    config = {'row': 4, 'col': 4, 'num': 4,
              'sub': 4, 'fix': 8,
              'method': method, 'num_reads': 100}
    assert method in ['hybrid']
    return config