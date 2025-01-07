'''
---------------------------------------------------------------------------------------
              _____  __     __ _____   _____  _      __  __
             |  __ \ \ \   / // ____| / ____|| |    |  \/  |
             | |__) | \ \_/ /| (___  | |     | |    | \  / |
             |  ___/   \   /  \___ \ | |     | |    | |\/| |
             | |        | |   ____) || |____ | |____| |  | |
             |_|        |_|  |_____/  \_____||______|_|  |_|
---------------------------------------------------------------------------------------
    A package for the contextualization of the logical DBN models corresponding to
                   signaling and gene regulatory systems
---------------------------------------------------------------------------------------

    Salma Bayoumi:           salma.ismail.hamed@gmail.com
                             salma.bayoumi.001@student.uni.lu
    Sébastien De Landtsheer: sebastien.delandtsheer@uni.lu
                             seb@delandtsheer.com
    Prof Thomas Sauter:      thomas.sauter@uni.lu

---------------------------------------------------------------------------------------
    University of Luxembourg 2025 GPLv3
---------------------------------------------------------------------------------------
'''

# 1. Importing the required class from the PYPGLM module:
# 1.1 ProbabilisticGraphicalLogicalModel is a class used for probabilistic graphical models:
from PYPGLM.pypglm import ProbabilisticGraphicalLogicalModel
# 1.2 Required function for integrating multiple cell lines into onr globalized setting
from PYPGLM.pypglm import create_a_global_model, create_a_global_data
# 1.3 Importing the os module, which provides a way of using operating system-dependent functionality:
import os
# 1.4 Importing the pandas and NumPy libraries and aliasing them as 'pd' and 'np:
import pandas as pd
import numpy as np

# 2. Building a globalized network topology model for all the layers (e.g., cell lines)
#    across the investigated system, using create_a_global_model method, where the user
#    should provide:
#    the input file (e.g., the network topology model) through input_file argument,
#    suffix list (e.g., the suffixes of all the cell lines to be combined in a list) through suffix_list argument,
#    output file path (the desired path to save the globalized model) through output_file_path argument.

input_file = "data/legacy/DelMistro2018/crosstalk_model_final.xlsx"
suffix_list = ['parental', 'izi_cond']
output_file_path = "data/legacy/DelMistro2018/DelMistro_Global_model.xlsx"

create_a_global_model(input_file, suffix_list, output_file_path)


# 3. Building a globalized experimental data for all the layers (e.g., cell lines)
#    across the investigated system, using create_a_global_data method, where the user
#    should provide:
#    the input node values (e.g., the input node values sheet in the experimental data) through input_df argument,
#    the list of all output node values (e.g., the output node values sheet in each experimental data corresponding to each cell line) through list_data_outputs argument,
#    the list of all experimental errors (e.g., the error sheet in each experimental data corresponding to each cell line) through list_data_errors argument,
#    suffix list (e.g., the suffixes of all the cell lines to be combined in a list) through suffix_list argument,
#    global data file path (the desired path to save the globalized data) through global_data_path argument.


input_df = pd.read_excel("data/legacy/DelMistro2018/crosstalk_parental_16h.xlsx", sheet_name='input')
output_df_cell_line1 = pd.read_excel("data/legacy/DelMistro2018/crosstalk_parental_16h.xlsx", sheet_name='output')
output_df_cell_line2 = pd.read_excel("data/legacy/DelMistro2018/crosstalk_izi_cond_16h.xlsx", sheet_name='output')
list_data_outputs = [output_df_cell_line1, output_df_cell_line2]
error_df_cell_line1 = pd.read_excel("data/legacy/DelMistro2018/crosstalk_parental_16h.xlsx", sheet_name='error')
error_df_cell_line2 = pd.read_excel("data/legacy/DelMistro2018/crosstalk_izi_cond_16h.xlsx", sheet_name='error')
list_data_errors = [error_df_cell_line1, error_df_cell_line2]
suffix_list = ['parental', 'izi_cond']
global_data_path = "data/legacy/DelMistro2018/DelMistro_combined_data.xlsx"

create_a_global_data(input_df, list_data_outputs, list_data_errors, suffix_list, global_data_path)


# 4. Integrating the globalized network with its associated globalized experimental data:
# 4.1 Loading the network topology model (e.g., DelMistro model) from the specified file path:
test_model = os.path.join("data/legacy/DelMistro2018/DelMistro_Global_model.xlsx")

# 4.2 Initializing the Probabilistic Graphical Logical Model instance:
pglm = ProbabilisticGraphicalLogicalModel()

# 4.3 Loading the network topology model into the pglm instance using the 'load_network' method:
pglm.load_network(test_model)

# 4.4 Loading the associated experimental data for the network model from the specified Excel file:
pglm.load_data("data/legacy/DelMistro2018/DelMistro_combined_data.xlsx")

# 5. Sanity checks (validating the Probabilistic Graphical Logical Model building):
pglm.check_network_consistency()

# 6. Optimizing the Probabilistic Graphical Logical Model using a specific type of
#    regularization (e.g., L1Groups) across a specified regularization strength (Lambda)
#    (e.g., -8), using pglm.gradient_descent() method, which
#    returns the following:
#   - "parameters": The optimal values of the parameters after optimization.
#   - "MSE": The Mean Squared Error of the final model.
#   - "regularization_cost": Cost associated with regularization if used.
#   - "total_cost": The total cost (MSE + regularization cost) if regularization is used.
#   - "BIC": Bayesian Information Criterion to evaluate the goodness of fit if regularization is used.
#   - "num_parameters": Number of BIC parameters with a standard deviation < 0.01 (if using L1 groups regularization).
#   - "time": Time taken to complete the optimization process.
#     User Options:
#   - Initialization Option: Users can select an initialization distribution from:
#     ['uniform', 'xavier', 'he'] using the "initialization_option" argument.
#   - Optimization Method: The user can choose their optimization method from:
#     ['SLSQP', 'L-BFGS-B', 'trust-constr'] using the "method" argument.
#   - Regularization Type: Specify the regularization type using the "reg_type" argument.
#     Examples include 'L1', 'L2', or 'L1Groups'.
#   - Regularization Strength: The user can adjust the regularization strength using "reg_lambda".
#   - The list of suffixes for conditions or experiments that require regularization
#     is specified by the "suffixes" argument.
#     By default, regularization is off. To enable it, the user must specify a value for "reg_lambda",
#     and the preferred type of regularization.

suffixes = ['_parental', '_izi_cond']
pglm.gradient_descent(initialization_option='xavier', method='SLSQP', reg_type='L1Groups', reg_lambda=2.0 ** -8, suffixes=suffixes)

# 7. To evaluate the regularization metrics across a specific range of regularization strength
#    (e.g., from -12 to -2 with increments of 0.5), use regularization_metrics_analysis() method,
#    which requires the following parameters:
#   - starting_reg_level: The starting value of the regularization strength (Lambda).
#   - ending_reg_level: The ending value of the regularization strength (Lambda).
#   - step_reg: The step size used to increment the regularization strength (Lambda).
#   - Initialization Option: Users can select an initialization distribution from:
#     ['uniform', 'xavier', 'he'] using the "initialization_option" argument.
#   - Optimization Method: The user can choose their optimization method from:
#     ['SLSQP', 'L-BFGS-B', 'trust-constr'] using the "method" argument.
#   - Regularization Type: Specify the regularization type using the "reg_type" argument.
#     Examples include 'L1', 'L2', or 'L1Groups'.
#   - The list of suffixes for conditions or experiments that require regularization
#     is specified by the "suffixes" argument.

#     This method returns a dataframe that include the following:
#   - 'Lambda': The lambda value used in the evaluation.
#   - 'BIC': The Bayesian Information Criterion value for the evaluation.
#   - 'MSE': The Mean Squared Error value.
#   - 'Regularization_Cost': The cost due to regularization.
#   - 'Total_Cost': The total cost combining the model's fitting cost and regularization.
#   - 'Parameters': The number of parameters used in the model.
#   - 'Time': The time taken for the evaluation (can be None if not available).
#   - 'Number_of_Parameters': The number of BIC parameters used, can be None if not available.

#     The output is accompanied by plots for each metric.


suffixes = ['_parental', '_izi_cond']

pglm.regularization_metrics_analysis(starting_reg_level=-12, ending_reg_level=-1.5, step_reg=0.5,
    initialization_option='xavier',
    method='SLSQP',
    reg_type='L1Groups',
    suffixes=suffixes
)





'''
            End of the script
'''


