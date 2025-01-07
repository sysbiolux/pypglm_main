import numpy as np
import networkx as nx
import pandas as pd


# New function to return each dictionary that exist in
# the main dictionary of the network which matches
# the names of standard dictionaries that exist in a separate list
#  and then this function returns only a dictionary that include  all the other dictionaries.
# that match the names of items in the given list
# other way, it will raise an error
def create_specific_dict(main_dict, item_list):
    specific_dicts = {}
    for key, value in main_dict.items():
        if key in item_list:
            specific_dicts[key] = value
        else:
            print(
                "The names of all dictionaries of the main dictionary in the network don't match the standard names of dictionaries"
            )
    return specific_dicts


# a function to convert Nan into 1 for the checking if all nodes
# (except inputs) have at least one incoming positive edge so Nan can't be Zero
def convert_Nan_to_1(numpy_array):
    return np.where(np.isnan(numpy_array), 1, numpy_array)


# a function to globalize the network files:

def create_a_global_model(input_file, suffix_list, output_file_path):

    model_df = pd.read_excel(input_file)
    expanded_df = pd.DataFrame(columns=model_df.columns)

    for _, row in model_df.iterrows():
        for suffix in suffix_list:
            new_row = row.copy()
            new_row['Output'] = f"{row['Output']}_{suffix}"
            new_row['parameter'] = f"{row['parameter']}_{suffix}"
            if row['Input'] in model_df['Output'].values:
                new_row['Input'] = f"{row['Input']}_{suffix}"
            expanded_df = pd.concat([expanded_df, pd.DataFrame([new_row])], ignore_index=True)

    expanded_df.reset_index(drop=True, inplace=True)
    expanded_df.to_excel(output_file_path, index=False)
    print(f"Expanded DataFrame saved to '{output_file_path}'.")


# a function to globalize the experimental data files:

def create_a_global_data(input_df, list_data_outputs, list_data_errors, suffix_list, global_data_path):
    def add_suffixes_and_concatenate(dataframes, suffixes):
        for df, suffix in zip(dataframes, suffixes):
            new_columns = [df.columns[0]] + [f"{col}_{suffix}" for col in df.columns[1:]]
            df.columns = new_columns
        result_df = dataframes[0]

        for df in dataframes[1:]:
            df_no_annotation = df.drop(columns=['Annotation'])
            result_df = pd.concat([result_df, df_no_annotation], axis=1)

        return result_df
    result_df_outputs = add_suffixes_and_concatenate(list_data_outputs, suffix_list)
    result_df_error = add_suffixes_and_concatenate(list_data_errors, suffix_list)
    with pd.ExcelWriter(global_data_path, engine='openpyxl') as writer:
        if input_df is not None:
            input_df.to_excel(writer, sheet_name='input', index=False)
        result_df_outputs.to_excel(writer, sheet_name='output', index=False)
        result_df_error.to_excel(writer, sheet_name='error', index=False)
    print(f"DataFrames have been saved to '{global_data_path}' with sheets 'input', 'output', and 'error'.")