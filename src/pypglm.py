import os
import sys
import glob
import pandas as pd
import warnings
import time
import pickle
import math
from scipy.optimize import minimize, LinearConstraint, Bounds
from src.utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


class ProbabilisticGraphicalLogicalModel:
    def __init__(self, network_file=None, data_files=None):

        self.node_values_history = {}
        self.fitted = False
        self.node_values = np.empty((0,))
        self.mse = None
        self.param_init_option = "uniform"
        self.network = None
        self.matdict = None
        self.params_names = None
        self.count_Np_Ni_params = None
        self.Nan_dict = None
        self.A_eq_vector = None
        self.A_ineq_vector = None
        self.lb_Ni_constranis = None
        self.ub_Np_constranis = None
        self.optimization_matrix = None
        self.data = None
        self.n_datapoints = None
        self.inputs = None
        self.outputs = None
        self.input_nodes_indices = None
        self.output_nodes_indices = None

        if network_file is not None:
            self.load_network(network_file)
        if data_files is not None:
            self.load_data(data_files)

    def __repr__(self):
        return f"{', '.join([f'{name}: {value}' for name, value in zip(list(self.network.nodes), self.node_values)])}"

    def __eq__(self, other):
        if isinstance(other, ProbabilisticGraphicalLogicalModel):
            if self.network.nodes == other.network.nodes:
                if self.matdict == other.matdict:
                    if self.node_values == other.node_values:
                        return True
        return False

    def load_network(self, network_file):
        _, file_extension = os.path.splitext(network_file)

        if file_extension.lower() == ".csv":
            network_data = pd.read_csv(network_file)
            self.network = self._process_network_data(network_data)
        elif file_extension.lower() in [".xls", ".xlsx"]:
            network_data = pd.read_excel(network_file, engine="openpyxl")
            self.network = self._process_network_data(network_data)
        elif file_extension.lower() == ".sif":
            self.network = self._load_network_sif(network_file)
        else:
            raise ValueError(
                "Invalid file format. Supported formats are CSV, XLS, XLSX, and SIF."
            )

        self.matdict = self._get_adjacency_matrices()
        self.matdict = self._tweak_matrices()
        self.fitted = False
        self.params_names = self._params_names()
        self.count_Np_Ni_params = self._count_Np_Ni_params()
        self.Nan_dict = self._create_Nan_dict()
        self.A_eq_vector = self._A_eq_vector()
        self.A_ineq_vector = self._A_ineq_vector()
        self.lb_Ni_constranis = self._lb_Ni_constranis()
        self.ub_Np_constranis = self._ub_Np_constranis()
        self.optimization_matrix = self._optimization_matrix()

    def _load_network_sif(self, sif_file: str):
        network = nx.DiGraph()
        with open(sif_file, "r") as f:
            for line in f:
                input_node, reaction, output_node = line.strip().split("\t")
                network.add_edge(
                    input_node,
                    output_node,
                    reaction=reaction,
                    parameter=f"{input_node}-{output_node}",
                    gate="N",
                )

        return network

    def _process_network_data(self, network_data):
        network = nx.DiGraph()
        for index, row in network_data.iterrows():
            input_node = row["Input"]
            reaction = row["Reaction"]
            output_node = row["Output"]
            parameter = (
                row["parameter"]
                if not pd.isna(row["parameter"])
                else f"{input_node}-{output_node}"
            )
            gate = row["gate"]

            if not network.has_node(input_node):
                network.add_node(input_node)
            if not network.has_node(output_node):
                network.add_node(output_node)

            network.add_edge(
                input_node,
                output_node,
                reaction=reaction,
                parameter=parameter,
                gate=gate,
            )

        return network

    def load_data(self, file_path):
        if isinstance(file_path, str):
            data = self._load_data_excel(file_path)
            self.fitted = False
            self.mse = None
        elif isinstance(file_path, list) and len(file_path) == 3:
            data = self._load_data_csv(file_path)
            self.fitted = False
            self.mse = None
        else:
            raise ValueError(
                "Invalid file_path argument. It should be either a single file (Excel) or a list of three files (CSV)."
            )

        self.data = data
        self.n_datapoints = self._n_datapoints()
        self.inputs = self._inputs()
        self.outputs = self._outputs()
        self.input_nodes_indices = self._input_nodes_indices()
        self.output_nodes_indices = self._output_nodes_indices()

    def _load_data_excel(self, file_path):
        sheets = ["input", "output", "error"]
        data = {}

        try:
            for sheet in sheets:
                data[sheet] = pd.read_excel(file_path, sheet_name=sheet, index_col=0)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return {}
        except ValueError as e:
            print(f"Error: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

        return data

    def _load_data_csv(self, file_paths):
        keys = ["input", "output", "error"]
        data = {}

        try:
            for key, file_path in zip(keys, file_paths):
                data[key] = pd.read_csv(file_path, index_col=0)
        except FileNotFoundError:
            print(f"Error: One of the files specified does not exist.")
            return {}
        except pd.errors.EmptyDataError:
            print("Error: One of the files is empty.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

        return data

    def _get_adjacency_matrices(self):
        """Private method for getting the adjacency matrices and the param vector"""
        matdict = {}
        n_nodes = self.network.number_of_nodes()
        Np = np.zeros((n_nodes, n_nodes))
        A = np.zeros((n_nodes, n_nodes))
        O = np.zeros((n_nodes, n_nodes))
        Ni = np.zeros((n_nodes, n_nodes))
        params = {
            "name": dict(),
            "matrix": dict(),
            "coord": dict(),
            "value": dict(),
        }

        for target_node_nr, target_node in enumerate(self.network):
            incoming_edges = self.network.in_edges(target_node, data=True)
            if len(incoming_edges) > 0:
                for source_node_nr, source_node in enumerate(self.network):
                    if self.network.has_edge(source_node, target_node):
                        edge_data = self.network.get_edge_data(source_node, target_node)
                        # basic edges
                        if edge_data["reaction"] == "->" and edge_data["gate"] == "N":
                            param_name = edge_data["parameter"]
                            if isinstance(param_name, str):
                                if len(incoming_edges) > 1:
                                    Np[target_node_nr][source_node_nr] = np.nan
                                    params["coord"][param_name] = (
                                        target_node_nr,
                                        source_node_nr,
                                    )
                                    params["name"][param_name] = param_name
                                    params["value"][param_name] = np.nan
                                    params["matrix"][param_name] = "Np"
                                else:
                                    Np[target_node_nr][source_node_nr] = 1
                            elif isinstance(param_name, (int, float)):
                                if len(incoming_edges) > 1:
                                    Np[target_node_nr][source_node_nr] = param_name
                                else:
                                    Np[target_node_nr][source_node_nr] = 1
                        # inhibitory edges
                        if edge_data["reaction"] == "-|" and edge_data["gate"] == "N":
                            param_name = edge_data["parameter"]
                            if isinstance(param_name, str):
                                Ni[target_node_nr][source_node_nr] = np.nan
                                # params[param_name] = (source_node_nr, target_node_nr)
                                params["coord"][param_name] = (
                                    target_node_nr,
                                    source_node_nr,
                                )
                                params["name"][param_name] = param_name
                                params["value"][param_name] = np.nan
                                params["matrix"][param_name] = "Ni"
                            elif isinstance(param_name, (int, float)):
                                Ni[target_node_nr][source_node_nr] = param_name

                        # AND gates
                        if edge_data["reaction"] == "->" and edge_data["gate"] == "A":
                            param_name = edge_data["parameter"]
                            if isinstance(param_name, str):
                                A[target_node_nr][source_node_nr] = 1
                                # params[param_name] = (source_node_nr, target_node_nr)
                                params["coord"][param_name] = (
                                    target_node_nr,
                                    source_node_nr,
                                )
                                params["name"][param_name] = param_name
                                params["value"][param_name] = 1
                                params["matrix"][param_name] = "A"
                            elif isinstance(param_name, (int, float)):
                                print(
                                    f'This is not allowed: node {list(self.network.nodes)[target_node_nr]} is Boolean but has a weight. It has been replaced by "1"'
                                )
                                A[target_node_nr][source_node_nr] = 1

                        # OR gates
                        if edge_data["reaction"] == "->" and edge_data["gate"] == "O":
                            param_name = edge_data["parameter"]
                            if isinstance(param_name, str):
                                O[target_node_nr][source_node_nr] = 1
                                # params[param_name] = (source_node_nr, target_node_nr)
                                params["coord"][param_name] = (
                                    target_node_nr,
                                    source_node_nr,
                                )
                                params["name"][param_name] = param_name
                                params["value"][param_name] = 1
                                params["matrix"][param_name] = "O"
                            elif isinstance(param_name, (int, float)):
                                print(
                                    f'This is not allowed: node {list(self.network.nodes)[target_node_nr]} is Boolean but has a weight. It has been replaced by "1"'
                                )
                                O[target_node_nr][source_node_nr] = 1

        matdict["Np"] = Np
        matdict["Ni"] = Ni
        matdict["A"] = A
        matdict["O"] = O
        matdict["params"] = params

        return matdict

    def _tweak_matrices(self):
        """
        Private method for adapting the matrices to the problem
        """
        remove = []

        matdict = self.matdict
        array = matdict["Np"]
        for i in range(array.shape[0]):
            column = array[i, :]
            nan_count = np.sum(np.isnan(column))
            if nan_count == 1:
                nan_index = np.where(np.isnan(column))
                nanarg_index = np.argwhere(np.isnan(column))
                # print(nan_index, nanarg_index[0])
                array[i, nan_index] = 1
                remove.append((i, nanarg_index[0]))
        matdict["Np"] = array
        # print(f'removed: {remove}')

        to_pop = []

        for param in matdict["params"]["name"].keys():
            if matdict["params"]["matrix"][param] == "Np":
                if matdict["params"]["coord"][param] in remove:
                    to_pop.append(param)

        for this_param in to_pop:
            matdict["params"]["coord"].pop(this_param)
            matdict["params"]["matrix"].pop(this_param)
            matdict["params"]["name"].pop(this_param)
            matdict["params"]["value"].pop(this_param)

        return matdict

    def check_network_consistency(self):
        """
        A method to Check if the network structure and the data files are consistent mathematically e.g.:
        - sum of nodes cannot be above 1
        - inputs need to be fixed in the data file
        - no self-loops
        - single component network

        return: True if everything is ok
        """

        # check 1: (all nodes without predecessors are in the inputs)
        nodes_without_predecessors = set(
            [
                node
                for node in self.network.nodes()
                if len(list(self.network.predecessors(node))) == 0
            ]
        )
        input_data_nodes = set(self.data["input"].columns)
        check1_difference = nodes_without_predecessors.difference(input_data_nodes)
        check1 = not check1_difference

        # check 2: (sum of activating edges is <=1)
        data_input_nodes = self.data["input"].columns.tolist()
        nodes_list = list(self.network.nodes())
        nodes_not_inputs_indices = [
            i for i, x in enumerate(nodes_list) if x not in data_input_nodes
        ]

        main_dictionary = create_specific_dict(
            self.matdict, ["Np", "Ni", "A", "O", "params"]
        )

        Np_array = main_dictionary["Np"]
        nodes_not_inputs_rows = Np_array[nodes_not_inputs_indices, :]
        row_sums_np = np.nansum(nodes_not_inputs_rows, axis=1)
        check2 = np.all(row_sums_np <= 1)

        # check 3: (sum of inhibitory edges is <=1)
        Ni_array = main_dictionary["Ni"]
        nodes_not_inputs_rows = Ni_array[nodes_not_inputs_indices]
        row_sums_ni = np.nansum(nodes_not_inputs_rows, axis=1)
        check3 = np.all(row_sums_ni <= 1)

        # check 4: (all non-inputs have at least one activating edge)
        new_desired_array = self.matdict["Np"] + self.matdict["A"] + self.matdict["O"]
        array_with_all_matrices = new_desired_array[nodes_not_inputs_indices]
        # converting all Nan into 1 using convert_Nan_to_1() function
        row_with_Nan_1 = convert_Nan_to_1(array_with_all_matrices)
        row_sums_with_Nan_1 = np.sum(row_with_Nan_1, axis=1)
        check4 = np.all(row_sums_with_Nan_1 > 0)

        # check 5: (all nodes without successors are outputs)
        nodes_without_successors = set(
            [
                node
                for node in self.network.nodes()
                if len(list(self.network.successors(node))) == 0
            ]
        )
        output_data_nodes = set(self.data["output"].columns)

        # Check if nodes_without_successors are in the outputs from the experimental data
        is_subset = nodes_without_successors.issubset(output_data_nodes)

        # finding nodes in nodes_without_successors that are not in the outputs from the experimental data
        nodes_not_in_output_data_nodes = nodes_without_successors - output_data_nodes
        check5 = is_subset

        # check 6: (there are no self-loops)
        self_loop_list = list(nx.nodes_with_selfloops(self.network))
        check6 = not self_loop_list

        # check 7: (there is only one component)
        number_of_components = nx.number_connected_components(
            nx.to_undirected(self.network)
        )
        check7 = number_of_components == 1

        # output:
        checks = [
            (
                check1,
                "All nodes are either defined or have a predecessor",
                "some undefined nodes do not have a predecessor",
                check1_difference,
            ),
            (
                check2,
                "The minimal value for sum of positive edges in Normal positive matrix is below 1 for all nodes",
                "some nodes have the sum of incoming positive edges already above 1",
                row_sums_np > 1,
            ),
            (
                check3,
                "The minimal value for sum of positive edges in Normal negative matrix is below 1 for all nodes",
                "some nodes have the sum of incoming inhibitory edges already above 1",
                row_sums_ni > 1,
            ),
            (
                check4,
                "All nodes (except inputs) have at least one incoming positive edge",
                "There exist nodes without at least one incoming positive edge",
                None,
            ),
            (
                check5,
                "All nodes without successor are mapped to the measured outputs",
                "Some nodes without successor are not mapped to the measured outputs",
                nodes_not_in_output_data_nodes,
            ),
            (
                check6,
                "There are no self-loops",
                "There are self-loops in the network",
                self_loop_list,
            ),
            (
                check7,
                "There is only one component",
                "There is more than one component",
                None,
            ),
        ]

        for check, success_msg, error_msg, error_detail in checks:
            if check:
                print(f"OK: {success_msg}")
            else:
                print(f"ERROR: {error_msg}")
                if error_detail is not None:
                    print(f"check nodes: {error_detail}")

        if all(check for check, *_ in checks):
            print("...All good!")
            return True
        else:
            print("...Modify the input files before continuing...")
            return False

    def initialize_params(self, option, seed=42):
        """
        A method for initializing the parameter values
        """
        rng = np.random.RandomState(seed)
        n_params = self._count_params()

        if option == "uniform":
            Np_indices = self._get_indices("Np")
            Ni_indices = self._get_indices("Ni")

            # Updating the parameters with randomized values for Np and Ni
            params = {**{key: rng.random() for key in Np_indices + Ni_indices}}

            # Normalizing the Np values
            self._normalize_Np()

        elif option in ["xavier", "he"]:
            scaled = self._xavier_he_init(option, n_params, rng)
            params = self._update_params_scaled(scaled)

        else:
            raise ValueError(f"Invalid option: {option}")

        return params

    def _count_Np_Ni_params(self):
        """
        Private method for obtaining the count of the parameters of the Np and Ni matrices
        """
        param_count = sum(1 for value in self.matdict["params"]["matrix"].values()
                          if value in ["Np", "Ni"])
        return param_count

    def _count_params(self):
        """
        Private method for obtaining the total parameters count
        """
        return sum(len(v) for v in self.matdict["params"]["coord"].values())

    def _params_names(self):
        """
        Private method for obtaining the names of the parameters
        """
        parameter_names = sorted(
            [key for key, value in self.matdict["params"]["matrix"].items() if value in ("Np", "Ni")],
            key=lambda key: (self.matdict["params"]["coord"][key][0], self.matdict["params"]["coord"][key][1])
        )
        return parameter_names

    def _get_indices(self, matrix_type):
        """
        Private method for obtaining the indices according to the chosen matrix type
        """
        return [key for key, value in self.matdict["params"]["matrix"].items() if value == matrix_type]

    def _normalize_Np(self):
        """
        Private method for normalizing the Np matrix
        """
        for i, row in enumerate(self.matdict["Np"]):
            row_sum = np.nansum(row)
            if row_sum > 1e-9:  # Checking if the row sum is non-zero
                self.matdict["Np"][i] /= row_sum

    def _xavier_he_init(self, option, n_params, rng):
        """
        Private method for initializing the parameters with either Xavier or He distribution
        """
        if option == "xavier":
            return rng.normal(0.5, 0.167, size=(n_params))
        elif option == "he":
            return np.zeros(n_params) + (rng.rand(n_params) / 1000000)

    def _update_params_scaled(self, scaled):
        """
        Private method for updating the scaled parameters
        """
        names_of_parameters = self.params_names
        return dict(zip(names_of_parameters, scaled))

    def _map_params(self, params_from_gradient_descent):
        """
        Private method for mapping the parameters to their respective matrices
        """
        if isinstance(params_from_gradient_descent, dict):
            for key, value in params_from_gradient_descent.items():
                if key in self.matdict["params"]["matrix"]:
                    matrix = self.matdict["params"]["matrix"][key]
                    coord = self.matdict["params"]["coord"][key]
                    self.matdict["params"]["value"][key] = value
                    self.matdict[matrix][coord[0], coord[1]] = value
        else:
            dictionary_parameters = self.make_dict(params_from_gradient_descent)
            for key, value in dictionary_parameters.items():
                if key in self.matdict["params"]["matrix"]:
                    matrix = self.matdict["params"]["matrix"][key]
                    coord = self.matdict["params"]["coord"][key]
                    self.matdict["params"]["value"][key] = value
                    self.matdict[matrix][coord[0], coord[1]] = value

    def make_dict(self, params_from_gradient_descent):
        """
        Private method for creating a dictionary for the gradient descent parameters
        """
        keys = [k for k, v in self.matdict["params"]["value"].items() if isinstance(v, float) and math.isnan(v)]
        return dict(zip(keys, params_from_gradient_descent))

    def _randomize_nodes(self, seed=42, lower_bound=0.0, upper_bound=1.0):
        """
        Private method for randomizing the node values
        """
        rng = np.random.RandomState(seed)
        n_nodes = self.network.number_of_nodes()
        self.node_values = rng.random((n_nodes,)) * (upper_bound - lower_bound) + lower_bound

    def _inputs(self):
        """
        Private method for obtaining the values of the inputs
        """
        inputs = self.data["input"].values
        return inputs

    def _outputs(self):
        """
        Private method for obtaining the values of the outputs
        """
        outputs = self.data["output"].replace('NaN', np.nan).values
        return outputs

    def _input_nodes_indices(self):
        """
        Private method for obtaining the input node indices
        """
        nodes = list(self.network.nodes)
        input_nodes_indices = np.array([nodes.index(name) for name in self.data["input"].columns])
        return input_nodes_indices

    def _output_nodes_indices(self):
        """
        Private method for obtaining the output node indices
        """
        nodes = list(self.network.nodes)
        output_nodes_indices = np.array([nodes.index(name) for name in self.data["output"].columns])
        return output_nodes_indices

    def _update_network(self):
        """
        Private method for obtaining the total parameters count
        """

        input_values = self.node_values[self.input_nodes_indices]

        new_node_values = np.matmul(self.matdict["Np"], self.node_values)
        negative_contributions = np.matmul(self.matdict["Ni"], self.node_values)
        new_node_values *= (1 - negative_contributions)

        A = self.matdict["A"]
        O = self.matdict["O"]

        idx_a = np.where(np.any(A == 1, axis=1))[0]
        idx_o = np.where(np.any(O == 1, axis=1))[0]

        for a_node_idx in idx_a:
            idx_in = np.where(A[a_node_idx] == 1)[0]
            new_node_values[a_node_idx] = np.prod(self.node_values[idx_in])

        for o_node_idx in idx_o:
            idx_in = np.where(O[o_node_idx] == 1)[0]
            new_node_values[o_node_idx] = 1 - np.prod(1 - self.node_values[idx_in])

        new_node_values[self.input_nodes_indices] = input_values

        diff = np.linalg.norm(new_node_values - self.node_values)
        self.node_values = new_node_values

        return float(diff)

    def _n_datapoints(self):
        """
        Private method for obtaining the number of datapoints
        """
        outputs = self.data["output"].replace('NaN', np.nan).values
        n_datapoints = np.isfinite(outputs).sum()
        return n_datapoints

    def calculate_regularization(self, sum_abs_diffs, reg_type='none', reg_lambda=0):
        """
        A method for calculating the regularization
        """
        if reg_type == 'none':
            return 0
        elif reg_type == 'L1':
            return reg_lambda * np.sum(np.abs(sum_abs_diffs))
        elif reg_type == 'L2':
            return reg_lambda * np.sum(np.square(sum_abs_diffs))
        elif reg_type == 'L1Groups':
            return reg_lambda * sum_abs_diffs
        else:
            raise ValueError("Unsupported regularization type. Choose 'none', 'L1', or 'L2'.")

    def simulate(self, params_gradient_descent,
                 tolerance=1e-8, max_iterations=1000,
                 reg_type='none', reg_lambda=0,
                 suffixes=None):
        """
        A method for simulating the non-input nodes
        """

        if suffixes is None:
            suffixes = []

        # Converting tolerance to float if necessary
        if not isinstance(tolerance, (float, int)):
            raise ValueError("tolerance must be a float or an integer")
        tolerance = float(tolerance)

        for i in self.params_names:
            self.matdict['params']['value'][i] = np.nan

        self._map_params(params_gradient_descent)

        error_sq = 0

        for input_row, output_row in zip(self.inputs, self.outputs):
            self.node_values[self.input_nodes_indices] = input_row
            diff = np.inf
            iteration = 0
            while diff > tolerance and iteration < max_iterations:
                diff = self._update_network()
                iteration += 1
            if iteration == max_iterations:
                print(
                    "(simulate) Warning: The difference could not reach less than the tolerance within the specified number of iterations.")

            simulated_values = self.node_values[self.output_nodes_indices]
            valid_indices = np.isfinite(output_row) & np.isfinite(simulated_values)
            error_sq += np.sum((output_row[valid_indices] - simulated_values[valid_indices]) ** 2)

        MSE = error_sq / self.n_datapoints

        # Calculating the absolute differences based on the current parameters
        param_values_by_suffix = {}
        if suffixes:  # If the suffixes are provided, the processing of the parameters is based on them
            for suffix in suffixes:
                param_values_by_suffix[suffix] = {key.replace(suffix, ''): value
                                                  for key, value in params_gradient_descent.items()
                                                  if suffix in key}

            all_keys = sorted(
                set().union(*[set(param_values.keys()) for param_values in param_values_by_suffix.values()]))

            # Creating the matrix based on the number of suffixes
            matrix_np = np.array([[param_values_by_suffix[suffix].get(key, 0) for suffix in suffixes]
                                  for key in all_keys])

            row_means = np.mean(matrix_np, axis=1)
            differences = np.abs(matrix_np - row_means[:, np.newaxis])
            abs_sum = np.array([np.sum(row) for row in differences])
            sum_abs_diffs = np.sum(abs_sum)

        else:
            # If no suffixes provided, skipping suffix-related logic
            sum_abs_diffs = 0

        # Calculating the regularization term
        regularization_term = self.calculate_regularization(sum_abs_diffs, reg_type, reg_lambda)

        # Combining the MSE with the regularization term
        total_cost = MSE + regularization_term

        return total_cost



    def _A_eq_vector(self):
        """
        Private method for computing the equality matrix
        """
        A_eq = np.where(np.isnan(self.matdict["Np"]), 1, 0)
        non_zeros_eq = np.any(A_eq != 0, axis=1)
        A_eq = A_eq[non_zeros_eq]
        A_eq_vector = [np.array(row[row == 1]) for row in A_eq]

        # Filter the arrays based on their length
        if any(len(arr) > 1 for arr in A_eq_vector):
            A_eq_vector = [arr for arr in A_eq_vector if len(arr) > 1]
        else:
            A_eq_vector = [arr for arr in A_eq_vector if len(arr) <= 1]

        return A_eq_vector

    def _A_ineq_vector(self):
        """
        Private method for computing the inequality matrix
        """
        A_ineq = np.where(np.isnan(self.matdict["Ni"]), 1, 0)
        non_zeros_ineq = np.any(A_ineq != 0, axis=1)
        A_ineq = A_ineq[non_zeros_ineq]
        A_ineq_vector = [np.array(row[row == 1]) for row in A_ineq]
        if any(len(arr) > 1 for arr in A_ineq_vector):
            A_ineq_vector = [arr for arr in A_ineq_vector if len(arr) > 1]
        else:
            A_ineq_vector = [arr for arr in A_ineq_vector if len(arr) <= 1]

        return A_ineq_vector

    def _optimization_matrix(self):
        """
        Private method for computing the optimization matrix
        """

        A_eq_vector = self.A_eq_vector
        A_ineq_vector = self.A_ineq_vector
        parameters_df = pd.DataFrame(index=self.network.adj.keys(), columns=self.network.adj.keys())
        for node in self.network.adj.keys():
            for neighbor, data in self.network.adj[node].items():
                parameters_df.at[node, neighbor] = data.get('parameter', None)
        parameters_df = parameters_df.fillna('')
        parameters_df = parameters_df.transpose()
        parameters_names = self.params_names
        all_nodes = list(self.network.nodes())
        all_nodes_dict = {
            index: [value for value in parameters_df.loc[index] if value in parameters_names and value != '']
            for index in all_nodes
        }
        only_nodes_with_count_of_params_dict = {key: len(value) for key, value in all_nodes_dict.items() if
                                                value and len(value) > 1}
        only_nodes_with_params_dict = {key: all_nodes_dict[key] for key in only_nodes_with_count_of_params_dict}
        Np_indices = [key for key, value in self.matdict["params"]["matrix"].items() if value == "Np"]
        Ni_indices = [key for key, value in self.matdict["params"]["matrix"].items() if value == "Ni"]
        Np_all_params_for_each_node = {
            key: [value for value in values if value in Np_indices and value != '']
            for key, values in only_nodes_with_params_dict.items()
            if len([value for value in values if value in Np_indices and value != '']) > 1
        }

        max_value_Np = sum(len(value_list) for value_list in Np_all_params_for_each_node.values())
        A_eq_matrix = [np.concatenate([row, np.zeros(max_value_Np - len(row))])
                       for row in A_eq_vector]
        A_eq_matrix = np.vstack(A_eq_matrix)
        A_eq_param_value_dict = {key: list(zip(Np_all_params_for_each_node[key], row)) for key, row in
                                 zip(Np_all_params_for_each_node.keys(), A_eq_matrix)}
        df_A_eq = pd.concat(
            [pd.DataFrame({key: dict(value)}, index=parameters_names) for key, value in A_eq_param_value_dict.items()],
            axis=1, sort=True).fillna(0)
        A_eq_df = df_A_eq.transpose().reindex(columns=parameters_names, fill_value=0)
        Equality_matrix = A_eq_df.values

        Ni_all_params_for_each_node = {
            key: [value for value in values if value in Ni_indices and value != '']
            for key, values in only_nodes_with_params_dict.items()
            if len([value for value in values if value in Ni_indices and value != '']) > 1
        }
        if not Ni_all_params_for_each_node:
            only_nodes_with_count_of_params_dict = {
                key: len(value) for key, value in all_nodes_dict.items() if value and len(value) >= 1
            }
            only_nodes_with_params_dict = {key: all_nodes_dict[key] for key in only_nodes_with_count_of_params_dict}
            Ni_all_params_for_each_node = {
                key: [value for value in values if value in Ni_indices and value != '']
                for key, values in only_nodes_with_params_dict.items()
                if len([value for value in values if value in Ni_indices and value != '']) == 1
            }
        max_value_Ni = sum(len(value_list) for value_list in Ni_all_params_for_each_node.values())
        A_ineq_matrix = [
            np.pad(row.astype(float), (max_value_Ni - len(row), 0), 'constant', constant_values=0.0)
            if max_value_Ni > 1 else row.astype(float)  # Condition to check max_value_Ni
            for row in A_ineq_vector
        ]
        A_ineq_param_value_dict = {key: list(zip(Ni_all_params_for_each_node[key], row)) for key, row in
                                   zip(Ni_all_params_for_each_node.keys(), A_ineq_matrix)}
        df_A_ineq = pd.concat(
            [pd.DataFrame({key: dict(value)}, index=parameters_names) for key, value in
             A_ineq_param_value_dict.items()],
            axis=1, sort=True).fillna(0)
        A_ineq_df = df_A_ineq.transpose().reindex(columns=parameters_names, fill_value=0)
        Inequality_matrix = A_ineq_df.values
        optimization_matrix = np.vstack((Equality_matrix, Inequality_matrix))

        return optimization_matrix

    def _lb_Ni_constranis(self):
        """
        Private method for computing the constraints of the Ni matrix
        """
        A_eq_vector = self.A_eq_vector
        A_ineq_vector = self.A_ineq_vector
        lb_v = list(np.ones(len(A_eq_vector))), list(np.zeros(len(A_ineq_vector)))
        lb_Ni_constranis = [i for j in lb_v for i in j]
        return lb_Ni_constranis

    def _ub_Np_constranis(self):
        """
        Private method for computing the constraints of the Np matrix
        """
        A_eq_vector = self.A_eq_vector
        A_ineq_vector = self.A_ineq_vector
        ub_v = list(np.ones(len(A_eq_vector))), list(np.ones(len(A_ineq_vector)))
        ub_Np_constranis = [i for j in ub_v for i in j]
        return ub_Np_constranis

    def _create_Nan_dict(self):
        """
        Private method for creating a NAN dictionary
        """
        Nan_list = [np.nan] * self.count_Np_Ni_params
        Nan_dict = dict(zip(self.params_names, Nan_list))
        return Nan_dict

    def gradient_descent(self, initialization_option=None, method=None,
                         reg_type='none', reg_lambda=0, suffixes=None):

        """
        A method for optimizing the network

        Return: optimal parameters
        """
        if suffixes is None:
            suffixes = []

        init_option = initialization_option if initialization_option is not None else "xavier"
        algo = method if method is not None else "SLSQP"
        warnings.filterwarnings("ignore")

        start_time = time.process_time()
        self._map_params(self.Nan_dict)

        optimization_matrix = self.optimization_matrix
        lb_Ni_constranis = self.lb_Ni_constranis
        ub_Np_constranis = self.ub_Np_constranis

        initialized_parameters = self.initialize_params(init_option)
        bounds = Bounds(np.zeros(self.count_Np_Ni_params), np.ones(self.count_Np_Ni_params))
        constraints = LinearConstraint(optimization_matrix, lb_Ni_constranis, ub_Np_constranis)

        self._randomize_nodes(seed=42)

        result = minimize(
            fun=lambda x: self.simulate(
                dict(zip(self.params_names, x)),
                tolerance=1e-8,
                max_iterations=1000,
                reg_type=reg_type,
                reg_lambda=reg_lambda,
                suffixes=suffixes  # Pass suffixes here
            ),
            x0=np.array(list(initialized_parameters.values())),
            method=algo,
            bounds=bounds,
            constraints=constraints,
        )

        end_time = time.process_time()
        elapsed_time = end_time - start_time

        params_gradient_descent = dict(zip(self.params_names, result.x))

        total_cost = result.fun

        #  handling multiple suffixes if provided
        param_values_by_suffix = {}
        if suffixes:
            for suffix in suffixes:
                param_values_by_suffix[suffix] = {key.replace(suffix, ''): value
                                                  for key, value in params_gradient_descent.items()
                                                  if suffix in key}
            all_keys = sorted(
                set().union(*[set(param_values.keys()) for param_values in param_values_by_suffix.values()]))

            # Creating the matrix dynamically based on the number of suffixes
            matrix_np = np.array([[param_values_by_suffix[suffix].get(key, 0) for suffix in suffixes]
                                  for key in all_keys])

            row_means = np.mean(matrix_np, axis=1)
            differences = np.abs(matrix_np - row_means[:, np.newaxis])

            abs_sum = np.array([np.sum(row) for row in differences])
            sum_abs_diffs = np.sum(abs_sum)

        else:
            sum_abs_diffs = 0

        # BIC calculation
        std_per_row = np.std(matrix_np, axis=1, ddof=1) if suffixes else np.array([])  # Avoid calculation if no suffixes
        rows_with_low_std = std_per_row < 0.01 if suffixes else np.array([])
        num_parameters = np.sum(rows_with_low_std) + matrix_np.shape[1] * np.sum(~rows_with_low_std) if suffixes else 0
        MSE = total_cost - self.calculate_regularization(sum_abs_diffs, reg_type, reg_lambda)

        # BIC formula: BIC = N * log(MSE) + num_parameters * log(N)
        BIC = self.n_datapoints * np.log(MSE) + num_parameters * np.log(self.n_datapoints) if suffixes else 0

        results = {
            "parameters": params_gradient_descent,
            "MSE": MSE,
            "regularization_cost": self.calculate_regularization(sum_abs_diffs, reg_type, reg_lambda),
            "total_cost": total_cost,
            "BIC": BIC,
            "num_parameters": num_parameters,
            "time": elapsed_time
        }

        return results

    def find_best_lambda(self, lambda_values, initialization_option=None, method=None, reg_type='none', suffixes=None):

        """
        A method for obtaining the best Regularization strength (Lambda)
        """

        best_lambda = None
        best_total_cost = float('inf')
        best_results = None
        best_bic = float('inf')
        best_bic_results = None
        all_results = []

        for reg_lambda in lambda_values:
            results = self.gradient_descent(
                initialization_option=initialization_option,
                method=method,
                reg_type=reg_type,
                reg_lambda=reg_lambda
                , suffixes=suffixes
            )

            MSE = results['MSE']
            regularization_cost = results['regularization_cost']
            total_cost = results['total_cost']
            BIC = results['BIC']

            all_results.append({'Lambda': reg_lambda,
                                "MSE": MSE, 'Regularization_Cost': regularization_cost,
                                'Total_Cost': total_cost, 'BIC': BIC})

            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_lambda = reg_lambda
                best_results = results

            if BIC < best_bic:
                best_bic = BIC
                best_bic_results = results

        return best_lambda, best_results, all_results, best_bic_results

    def evaluate_bic_for_lambda(self, lambda_values, initialization_option=None,
                                method=None, reg_type='none',
                                suffixes=None):

        """
        A method for obtaining the BIC for each Regularization strength (Lambda)
        """

        lambda_bics = []
        detailed_results = []
        num_parameters_list = []

        for reg_lambda in lambda_values:
            results = self.gradient_descent(
                initialization_option=initialization_option,
                method=method,
                reg_type=reg_type,
                reg_lambda=reg_lambda, suffixes=suffixes
            )

            BIC = results['BIC']
            MSE = results['MSE']
            regularization_cost = results['regularization_cost']
            total_cost = results['total_cost']
            params_gradient_descent = results['parameters']
            elapsed_time = results['time']
            num_parameters = results['num_parameters']

            lambda_bics.append((reg_lambda, BIC))

            detailed_results.append({
                'Lambda': reg_lambda,
                'BIC': BIC,
                "MSE": MSE,
                'Regularization_Cost': regularization_cost,
                'Total_Cost': total_cost,
                'Parameters': params_gradient_descent,
                'Time': elapsed_time,
                'Number_of_Parameters': num_parameters
            })

            num_parameters_list.append(num_parameters)

        return lambda_bics, detailed_results, num_parameters_list

    def fit(
            self,
            max_iterations=None,
            tolerance=None,
            initialization_option=None,
            equal_sum=None,
            seed=None,
    ):
        """
        A method for fitting the network
        """
        # Checks:
        assert self.network, "There is no network in this model"
        assert self.data, "There is no data in this model"

        max_iterations = max_iterations if max_iterations is not None else 1000
        tolerance = tolerance if tolerance is not None else 1e-8
        initialization_option = (
            initialization_option if initialization_option is not None else "uniform"
        )
        equal_sum = equal_sum if equal_sum is not None else True
        self.seed = seed if seed is not None else 42

        # Fit the model to the data
        results = self.gradient_descent(
            max_iterations=max_iterations,
            tolerance=tolerance,
            initialization_option=initialization_option,
            equal_sum=equal_sum,
            seed=seed,
        )

        mse = results["MSE"]
        best_params = results["parameters"]

        self.fitted = True
        self.mse = mse
        self.opt_params = best_params

    def predict(self, input_data):
        """
         A method for making predictions based on the input_data
        """
        pass

    def save_model(self, output_file: str):
        """
         A method for saving a complete model to file
        """
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        path = os.path.join("data", "saved_models")
        output_file = output_file.split(".")[0]
        ext = ".pglm"
        output_file = output_file + "." + ext

        try:
            file = open(os.path.join(path, output_file), "wb")
            pickle.dump(self, file)
        except FileNotFoundError:
            print("FileNotFoundError: The file does not exist.")
        except PermissionError:
            print(
                "PermissionError: You do not have the necessary permissions to access the file."
            )
        except OSError as e:
            print(f"OSError: Some OS error occurred. Details: {str(e)}")
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")

    def load_model(self, input_file):
        """
         A method for loading a complete model from file
        """
        input_file_raw = input_file.rsplit(".", 1)[0]
        path = os.path.join("data", "saved_models", input_file_raw)
        files = glob.glob(f"{path}.*")

        print(files)

        if not files:
            print("File does not exist")
        elif len(files) > 1:
            print("There are multiple files:")
            for file in files:
                print(file)
        else:
            filepath = files[0]

        try:
            file = open(filepath, "rb")
            return pickle.load(file)
        except FileNotFoundError:
            print("FileNotFoundError: The file does not exist.")
        except PermissionError:
            print(
                "PermissionError: You do not have the necessary permissions to access the file."
            )
        except OSError as e:
            print(f"OSError: Some OS error occurred. Details: {str(e)}")
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")

    def re_simulate_and_plot(self, initialization_option=None, method=None,
                             reg_type='none', reg_lambda=0,
                             tolerance=1e-8, max_iterations=1000,
                             suffixes=None):
        """
        re_simulate_and_plot function is used to get the final output node values
        after the optimization has finished, and plot:
        1- The correlation between simulated vs. observed node values.
        2- The node values at each experimental condition.
        """
        if suffixes is None:
            suffixes = []

        results = self.gradient_descent(
            initialization_option=initialization_option,
            method=method,
            reg_type=reg_type,
            reg_lambda=reg_lambda, suffixes=suffixes
        )

        params_gradient_descent = results['parameters']

        self._map_params(params_gradient_descent)

        all_simulated_outputs = []
        all_observed_outputs = []
        simulated_values_all_conditions = []

        SD = self.data['error'].fillna(0).values

        # Running the simulation once using the final parameters
        for input_row, observed_output in zip(self.inputs, self.outputs):
            self.node_values[self.input_nodes_indices] = input_row
            diff = np.inf
            iteration = 0

            # Re-running the network to convergence using the final parameters
            while diff > tolerance and iteration < max_iterations:
                diff = self._update_network()  # Updating network values
                iteration += 1

            # After convergence, capturing the final node values and output values
            simulated_output = self.node_values[self.output_nodes_indices].copy()

            # Storing simulated and observed values for plotting
            valid_indices = np.isfinite(observed_output) & np.isfinite(simulated_output)
            all_simulated_outputs.append(simulated_output[valid_indices])
            all_observed_outputs.append(observed_output[valid_indices])

            # Storing simulated values for all conditions for the molecular profiles plot
            simulated_values_all_conditions.append(simulated_output)

        # Converting lists to arrays for easier manipulation
        all_simulated_outputs = np.concatenate(all_simulated_outputs)
        all_observed_outputs = np.concatenate(all_observed_outputs)

        # Plot 1: Simulated vs Observed
        model = LinearRegression().fit(all_simulated_outputs.reshape(-1, 1), all_observed_outputs)
        intercept, slope = model.intercept_, model.coef_[0]
        r_squared = model.score(all_simulated_outputs.reshape(-1, 1), all_observed_outputs)
        p_value = stats.pearsonr(all_simulated_outputs, all_observed_outputs)[1]

        plt.figure(figsize=(6, 6))
        plt.scatter(all_observed_outputs, all_simulated_outputs, color='k', marker='.')
        x = np.array([0, 1])
        plt.plot(x, intercept + slope * x, '--k')

        plt.title(f'PYPGLM', fontsize=16, fontweight='bold')
        plt.xlabel("Observed", fontweight='bold', fontsize=15, labelpad=10)
        plt.ylabel("Simulated", fontweight='bold', fontsize=15, labelpad=10)
        plt.grid(True)
        plt.tick_params(axis='x', labelsize=11, width=2)
        plt.tick_params(axis='y', labelsize=11, width=2)
        plt.text(0.1, 0.92, f'$\\mathbf{{R^2 = {r_squared:.3f}, \\; p = {p_value:.3e}}}$',
                 fontsize=14, ha='left', va='center', transform=plt.gca().transAxes)
        plt.show()

        # Molecular Profiles (Experimental Condition Plot)
        num_conditions = len(self.inputs)
        num_outputs = simulated_values_all_conditions[0].size
        NLines = int(np.ceil(np.sqrt(num_outputs)))
        NCols = int(np.ceil(num_outputs / NLines))

        fig, axes = plt.subplots(NLines, NCols, figsize=(3, 4))
        axes = axes.flatten()
        simulated_values_all_conditions = np.array(simulated_values_all_conditions)

        legend_handles = []
        legend_labels = []

        def add_jitter(x, jitter_amount=0.05):
            return x + np.random.uniform(-jitter_amount, jitter_amount, size=x.shape)

        # Plotting each output state in a subplot
        for i in range(num_outputs):
            ax = axes[i]

            # Plotting experimental data first (in green) with error bars
            error_bar = ax.errorbar(
                np.arange(1, num_conditions + 1),
                self.outputs[:, i],
                yerr=SD[:, i],
                fmt='gs', markersize=2, linewidth=1, color=[0.4, 0.6, 0],
                label='Observed'
            )

            if i == 0:  # Add to legend only once
                legend_handles.append(error_bar[0])
                legend_labels.append('Observed')

            # Adding jitter to the x-coordinates for simulated data
            jittered_x = add_jitter(np.arange(1, num_conditions + 1))

            # Plotting simulated data on top with jitter
            sim_line, = ax.plot(
                jittered_x,
                simulated_values_all_conditions[:, i],
                'b.', markersize=20 / np.sqrt(num_outputs)
            )

            if i == 0:
                legend_handles.append(sim_line)
                legend_labels.append('PYPGLM Simulated')

            ax.set_xlim([0, num_conditions + 1])
            ax.set_ylim([0, 1.1])
            ax.set_xticks(np.arange(1, num_conditions + 1))
            ax.set_xticklabels(
                self.data['output'].index.values, fontsize=10, fontweight='bold', rotation=45
            )
            ax.set_yticks(np.linspace(0, 1.1, 6))
            ax.set_yticklabels(
                [f"{tick:.1f}" for tick in np.linspace(0, 1.1, 6)],
                fontsize=10, fontweight='bold'
            )
            ax.set_title(self.data['output'].columns.values[i], fontsize=12, fontweight='bold')

            if i >= num_outputs - NCols:  # This checks if subplot is in the last row
                ax.set_xlabel('Experimental Conditions', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', labelsize=10)
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.supylabel('Value', fontsize=13, fontweight='bold', x=0.08, y=0.55)

        legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            bbox_to_anchor=(0.26, .99),
            loc='upper left',
            fontsize=11,
            frameon=True, ncol=2
        )
        plt.setp(legend.get_texts(), fontweight='bold')  # Make legend font bold

        fig.tight_layout(rect=[0, 0, 0.1, 0.95])
        fig.subplots_adjust(bottom=0.2)

        plt.show()

        return all_simulated_outputs

    def regularization_metrics_analysis(self, starting_reg_level=None, ending_reg_level=None, step_reg=None,
                                        initialization_option=None,
                                        method=None, reg_type='none',
                                        suffixes=None):
        """
        A method for obtaining the regularization evaluation metrics
        (total cost, regularization cost, MSE, BIC and BIC parameter count) for
        each Regularization strength (Lambda).
        It generates plots for all these metrics.
        """

        PowerLambda = np.arange(starting_reg_level, ending_reg_level, step_reg)
        exponentiated_values = 2.0 ** PowerLambda
        ListLambda = np.insert(exponentiated_values, 0, 0)

        detailed_results = []

        # Running gradient descent for each lambda value
        for reg_lambda in ListLambda:
            results = self.gradient_descent(
                initialization_option=initialization_option,
                method=method,
                reg_type=reg_type,
                reg_lambda=reg_lambda, suffixes=suffixes
            )

            detailed_results.append({
                'Lambda': reg_lambda,
                'BIC': results['BIC'],
                'MSE': results['MSE'],
                'Regularization_Cost': results['regularization_cost'],
                'Total_Cost': results['total_cost'],
                'Parameters': results['parameters'],
                'Time': results.get('time', None),
                'Number_of_Parameters': results.get('num_parameters', None)
            })

        # Flattening the detailed_results into a DataFrame
        all_flattened_data = [
            {
                'Lambda': result.get('Lambda'),
                'BIC': result.get('BIC'),
                'MSE': result.get('MSE'),
                'Regularization_Cost': result.get('Regularization_Cost'),
                'Total_Cost': result.get('Total_Cost'),
                'Parameters': result.get('Parameters'),
                'Time': result.get('Time', None),
                'Number_of_Parameters': result.get('Number_of_Parameters', None)
            }
            for result in detailed_results
        ]

        self.df = pd.DataFrame(all_flattened_data)

        # Extracting variables for plotting
        formatted_lambdas = ['base'] + [f'{value:.8f}'.rstrip('0').rstrip('.') for value in PowerLambda]
        python_BIC = df['BIC']
        python_Nparams = df['Number_of_Parameters']
        python_MSE_log10 = np.log10(df['MSE'])
        python_regularization_cost = np.log10(df['Regularization_Cost'])
        python_total_cost = np.log10(df['Total_Cost'])

        # Finding minimum and second minimum indices
        def find_min_indices(arr):
            min_idx = np.argmin(arr)
            arr_with_inf = np.copy(arr)
            arr_with_inf[min_idx] = np.inf
            second_min_idx = np.argmin(arr_with_inf)
            return min_idx, second_min_idx

        min_python_idx, second_lowest_idx = find_min_indices(python_BIC)

        # Plot 1: BIC vs Lambda
        plt.figure(figsize=(10, 5))
        plt.plot(formatted_lambdas, python_BIC, linestyle='-', color='b', marker='o', label='PYPGLM')
        plt.plot(formatted_lambdas[min_python_idx], python_BIC[min_python_idx], '*', color='#fd1300', markersize=10)
        plt.xlabel('Lambda Values', fontweight='bold', fontsize=16)
        plt.ylabel('BIC', fontweight='bold', fontsize=16)
        plt.xticks(rotation=90, ha="right")
        plt.grid(True)
        plt.legend(fontsize=14, loc="best")
        plt.tight_layout()
        plt.show()

        # Plot 2: Number of Parameters vs Lambda
        plt.figure(figsize=(10, 5))
        plt.plot(formatted_lambdas, python_Nparams, linestyle='-', color='b', marker='o', label='PYPGLM')
        plt.xlabel('Lambda Values', fontweight='bold', fontsize=16)
        plt.ylabel('Number of Parameters', fontweight='bold', fontsize=16)
        plt.xticks(rotation=90, ha="right")
        plt.grid(True)
        plt.legend(fontsize=14, loc="best")
        plt.tight_layout()
        plt.show()

        # Plot 3: Log10 of MSE vs Lambda
        plt.figure(figsize=(10, 6))
        plt.plot(formatted_lambdas, python_MSE_log10, marker='o', linestyle='-', color='b', label='PYPGLM')
        plt.xlabel('Lambda Values', fontweight='bold', fontsize=16)
        plt.ylabel('Log10 of Mean Squared Error (MSE)', fontweight='bold', fontsize=16)
        plt.xticks(rotation=90, ha="right")
        plt.grid(True)
        plt.legend(fontsize=14, loc="best")
        plt.tight_layout()
        plt.show()

        # Plot 4: Log10 of Regularization Cost vs Lambda
        plt.figure(figsize=(10, 6))
        plt.plot(formatted_lambdas, python_regularization_cost, marker='o', linestyle='-', color='b', label='PYPGLM')
        plt.xlabel('Lambda Values', fontweight='bold', fontsize=16)
        plt.ylabel('Log10 of Regularization Cost', fontweight='bold', fontsize=16)
        plt.xticks(rotation=90, ha="right")
        plt.grid(True)
        plt.legend(fontsize=14, loc="best")
        plt.tight_layout()
        plt.show()

        # Plot 5: Log10 of Total Cost vs Lambda
        plt.figure(figsize=(10, 6))
        plt.plot(formatted_lambdas, python_total_cost, marker='o', linestyle='-', color='b', label='PYPGLM')
        plt.xlabel('Lambda Values', fontweight='bold', fontsize=16)
        plt.ylabel('Log10 of Total Cost', fontweight='bold', fontsize=16)
        plt.xticks(rotation=90, ha="right")
        plt.grid(True)
        plt.legend(fontsize=14, loc="best")
        plt.tight_layout()
        plt.show()

        return self.df



