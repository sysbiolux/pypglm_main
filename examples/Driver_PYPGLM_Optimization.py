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

# 1.2 Importing the os module, which provides a way of using operating system-dependent functionality:
import os

# 2. Integrating the network with its associated experimental data:
# 2.1 Loading the network topology model (e.g., PDGF model) from the specified file path:
#     The model is stored in an Excel file located in the 'data/legacy/' directory.
test_model = os.path.join("data/legacy/PDGF/PDGF_model.xlsx")

# 2.2 Initializing the Probabilistic Graphical Logical Model instance:
pglm = ProbabilisticGraphicalLogicalModel()

# 2.3 Loading the network topology model into the pglm instance using the 'load_network' method:
pglm.load_network(test_model)

# 2.4 Loading the associated experimental data for the network model from the specified Excel file:
pglm.load_data("data/legacy/PDGF/PDGF_meas.xlsx")

# 3. Sanity checks (validating the Probabilistic Graphical Logical Model building):
pglm.check_network_consistency()

# 4. Optimizing the Probabilistic Graphical Logical Model using pglm.gradient_descent() method, which
#    returns the following:
#   - "parameters": The optimal values of the parameters after optimization.
#   - "MSE": The Mean Squared Error of the final model.
#   - "regularization_cost": Cost associated with regularization if used.
#   - "total_cost": The total cost (MSE + regularization cost) if regularization is used.
#   - "BIC": Bayesian Information Criterion to evaluate the goodness of fit if regularization is used.
#   - "num_parameters": Number of BIC parameters with a standard deviation < 0.01 (if using L1 groups regularization).
#   - "time": Time taken to complete the optimization process.
#     User Options:
#   - Optimization Method: The user can choose their optimization method from:
#     ['SLSQP', 'L-BFGS-B', 'trust-constr'] using the "method" argument.
#   - Initialization Option: Users can select an initialization distribution from:
#     ['uniform', 'xavier', 'he'] using the "initialization_option" argument.
#   - Regularization Type: Specify the regularization type using the "reg_type" argument.
#     Examples include 'L1', 'L2', or 'L1Groups'.
#   - Regularization Strength: The user can adjust the regularization strength using "reg_lambda".
#   - The list of suffixes for conditions or experiments that require regularization
#     is specified by the "suffixes" argument.
#     By default, regularization is off. To enable it, the user must specify a value for "reg_lambda",
#     and the preferred type of regularization.

pglm.gradient_descent(initialization_option='xavier', method='SLSQP', reg_type='none', reg_lambda=0, suffixes=None)

# 5. To get the final output node values after the optimization has finished,
#    and the figures related to:
#    a. the correlation between simulated vs. observed node values.
#    b. the node values at each experimental condition.
pglm.re_simulate_and_plot()


'''
            End of the script
'''