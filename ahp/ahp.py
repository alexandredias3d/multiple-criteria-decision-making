#!/usr/bin/env python3
'''
    Author: Alexandre Henrique Teixeira Dias
    E-mail: alehenriquedias@gmail.com
    Created: 02/28/2018
    Last modified: 04/20/2018
    Description: This is a simple implementation of the 
                    Analytic Hierarchy Process. It
                    supports pairwise comparison matrix
                    (subjective weighting) as well as
                    entropy weight generation (objective
                    weighting).
    Python Version: 3.6


    Changelog:
    -04/10/2018:    Entropy Weight Generation.
    -04/11/2018:    Fixed the .dat file import.
    -04/20/2018:    Can compute the complement of input data.
'''

# Using numpy as scientific computing module.
import numpy as np

# Module to parse the command line arguments.
import argparse


# Labels to access the info stored in the variable problem
# formulation.
GOAL = 0
CRITERIA = 1
ALTERNATIVES = 2
COSTS = 3

# Import the hierarchy from the .info file and return the
# content in a tuple. This function gets the filepath as
# a parameter and return the formulation of the problem.
def import_info_file(filepath):

    raw_info = open(filepath, 'r')
    raw_info = raw_info.read().splitlines()
    
    # Separate the goal from the label goal 
    goal = raw_info[0].split(':')[1]
  
    # Get all the criteria and add them in a list.
    i = 1
    criteria_list = []
    criterion = raw_info[i].split(':')
    while criterion[0] == 'criterion':
        criteria_list.append(criterion[1])
        i += 1
        criterion = raw_info[i].split(':')
    
    # Get all the alternatives and add them in a list.
    alternatives_list = []
    alternative = criterion # The failed criterion is an alternative.
    while alternative[0] == 'alternative':
        alternatives_list.append(alternative[1])
        i += 1
        if i == len(raw_info):
            break
        alternative = raw_info[i].split(':')

    # Costs were already split at alternative, just get its value.
    costs = alternative[1]

    # Create a tuple containing all information from the problem.
    problem_formulation = (goal, criteria_list, alternatives_list, costs)

    return problem_formulation

# Import the priority file and returns a bidimensional array
# containing the values read.
def import_prty_file(filepath):
    
    criteria_pwcomp = np.loadtxt(filepath)
    return criteria_pwcomp

# Import the data file and returns a bidimensional array containing the
# data itself (must have one entry for each alternative and one column
# for each criterion) and the last line is the costs of each alternative
def import_dat_file(filepath, costs=False):

    if costs:
        # All lines but the last
        ratings = np.genfromtxt(filepath, skip_footer=1)
        n = ratings.shape[0] # number of alternatives / rows

        # Only the last line
        costs = np.genfromtxt(filepath, skip_header=n)

        return ratings, costs
    else:
        # All lines from .dat file
        return np.genfromtxt(filepath)

# Compute the approximattion of a eigenvector
def compute_eigenvector(array2d):
    
    array2d_squared = np.matmul(array2d, array2d) 
    rows_sum = np.sum(array2d_squared, axis=1)
    rows_sum_total = np.sum(rows_sum)
    eigenvector = rows_sum / rows_sum_total

    return (array2d_squared, eigenvector)

# Compute the criteria rankings using the approximation of
# the eigenvector method.
def compute_criteria_rankings(criteria):

    while True:
        criteria_squared, eigenvector = compute_eigenvector(criteria)

        new_criteria_squared, new_eigenvector = \
                                compute_eigenvector(criteria_squared)

        eigenvectors_difference = np.subtract(eigenvector, new_eigenvector)
        eigenvectors_difference = np.absolute(eigenvectors_difference)
        
        if all(value < 0.001 for value in eigenvectors_difference):

            return new_eigenvector

        criteria = new_criteria_squared
        eigenvector = new_eigenvector

# Compute the criteria rankings using entropy weight generation.
# It must receive a normalized decision matrix containing the
# alternatives.
def compute_weights_using_entropy(p):

    # From the paper:
    #   - p is the decision matrix (normalized_alternatives)
    #   - n is the number of criterion
    #   - m is the number of alternatives
    m = p.shape[0]
    n = p.shape[1]

    entropy_constant = 1/np.log(m)
    criteria_entropies = []

    # Compute all the entropies
    for i in range(0, n):
        plnps = [p[j, i] * np.log(p[j, i]) if p[j, i] != 0 else 0 \
                    for j in range(0,m)]

        # Compute the entropy for the ith parameter and append on the list
        entropy = -entropy_constant + sum(plnps)
        criteria_entropies.append(entropy)

    degree_of_diversification = [1 - criteria_entropies[i] for i in range(0, n)]
    sum_dod = sum(degree_of_diversification)

    # The weight is the degree of importance of the criterion, in AHP terms, it is
    # the criterion ranking
    weights = np.array([degree_of_diversification[i]/sum_dod for i in range(0, n)])

    return weights

# Compute the alternatives rankings in order to take a decision
# given the already computed data from the alternatives and the 
# criteria ranking.
def compute_alternatives_rankings(alternatives_info, criteria_ranking):
    
    alternatives_rankings = np.matmul(alternatives_info, criteria_ranking)
    return alternatives_rankings

# Normalize the columns of a given array
def normalize_columns(array):
    
    columns_sum = np.sum(array, axis=0)
    normalized_array = np.divide(array, columns_sum, out=np.zeros_like(array),
                                 where=columns_sum!=0)

    return normalized_array

# Print the results in a pleasant way.
def print_results(problem_formulation, alternatives_rankings, fancy=True):

    print('########## Results (fancy alternatives priorities) ##########')

    for i in range(0, len(alternatives_rankings)):
        ranking = str(round(alternatives_rankings[i] * 100, 4)) + '%' if \
                fancy else alternatives_rankings[i]
       
        print(problem_formulation[ALTERNATIVES][i], ranking)

# Print the benefit-cost ratio
def print_BCR(problem_formulation, benefit_cost_ratio):
 
    print('########## Results (benefit-cost ratio) ##########')

    for i in range(0, len(benefit_cost_ratio)):
        print(problem_formulation[ALTERNATIVES][i], benefit_cost_ratio[i])

# Print the raw array received as a parameter
def print_raw(array): 
    for i in range(0, array.shape[0]):
        print(int(round(array[i], 4) * 10000), end=('\n' if i == array.shape[0]-1 else '\t'))

def create_parser():
    # Using argparse to parse the command line arguments
    parser = argparse.ArgumentParser(description='Simple AHP Program')
    parser.add_argument('-i', '--input', type=str, help='problem name',
                        required=True)
    parser.add_argument('-e', '--entropy', help='entropy weight generation',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='detailed output', 
                        action='store_true')
    parser.add_argument('-c', '--complement', help='compute complement of data',
                        action='store_true')
    return parser

# Run this Python Script as a standalone program
if __name__ == '__main__':

    parser = create_parser()
    
    args = parser.parse_args()

    problem_formulation = import_info_file(args.input + '.info')
   
    # If Entropy Weight Generation is true, there is no need to read the
    # pairwise comparison matrix.
    criteria_pwcomp = None
    if not args.entropy: 
        criteria_pwcomp = import_prty_file(args.input + '.prty')
   
    # Read an normalize the data 
    alternatives_data = None
    alternatives_costs = None
    normalized_costs = None
    if problem_formulation[COSTS]:
        alternatives_data, alternatives_costs = \
            import_dat_file(args.input + '.dat', problem_formulation[COSTS])
        normalized_costs = normalize_columns(alternatives_costs.T)
    else:
        alternatives_data = \
            import_dat_file(args.input + '.dat', problem_formulation[COSTS])

    print(alternatives_data)
    if args.complement:
        for element in np.nditer(alternatives_data, op_flags=['readwrite']):
            element[...] = 100 - element
    print(alternatives_data)        

    normalized_alternatives = normalize_columns(alternatives_data)
    
    criteria_rankings = None
    if not args.entropy: 
        criteria_rankings = compute_criteria_rankings(criteria_pwcomp)
    else:
        criteria_rankings = compute_weights_using_entropy(normalized_alternatives)

    alternatives_rankings = compute_alternatives_rankings\
                    (normalized_alternatives, criteria_rankings)


    benefit_cost_ratio = np.divide(alternatives_rankings,
        normalized_costs, out=np.zeros_like(alternatives_rankings),
        where=normalized_costs!=0)
    print(normalized_costs)
    normalized_bcr = normalize_columns(benefit_cost_ratio.T)

    if args.verbose:
        print('######## Problem Formulation ########\n', problem_formulation)
        print('###### Criteria Pairwise Comparison ######\n', criteria_pwcomp)
        print('# Data and costs #\n', alternatives_data, alternatives_costs)
        print('### Normalized alternatives ###\n', normalized_alternatives) 
        print('########## Criteria rankings ##########\n', criteria_rankings)
        print('########## Alternatives rankings ##########\n', alternatives_rankings)
        print_results(problem_formulation, alternatives_rankings)
        print_BCR(problem_formulation, benefit_cost_ratio)
        print_BCR(problem_formulation, normalized_bcr)

    print_raw(normalized_bcr)
