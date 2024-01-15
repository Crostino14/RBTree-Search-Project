"""
Course: High Performance Computing 2023/2024

Lecturer: Francesco Moscato      fmoscato@unisa.it

Student and Creator:
Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it

Source Code for creating plots and tables of parallel and sequential version's performance data.

Copyright (C) 2023 - All Rights Reserved

This file is part of RB Tree Search Project.

This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RB Tree Search Project.
If not, see <http://www.gnu.org/licenses/>.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def format_version_name(dir_name):
    """
    Formats the version name based on the directory name.
    
    Args:
    - dir_name (str): The directory name.
    
    Returns:
    - str: Formatted version name.
    """
    if 'sequential' in dir_name.lower():
        return 'Sequential'
    elif 'mpi_openmp' in dir_name.lower():
        return 'MPI_OpenMP'
    elif 'cuda' in dir_name.lower():
        return 'CUDA_OpenMP'
    else:
        return 'Unknown'

def format_float(value):
    """
    Formats a floating-point value to have nine decimal places.
    
    Args:
    - value (float): The floating-point value.
    
    Returns:
    - str: Formatted value with nine decimal places.
    """
    return f"{value:.9f}"

def read_and_process_csv(file_path, seq_df=None):
    """
    Reads and processes a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file.
    - seq_df (pandas.DataFrame, optional): Sequential DataFrame. Defaults to None.
    
    Returns:
    - pandas.DataFrame: Processed DataFrame.
    """
    df = pd.read_csv(file_path)
    
    # Format floats with 9 decimal places
    for col in ['Search Time (s)', 'Total Program Time (s)']:
        df[col] = df[col].apply(format_float)

    # Remove the 'Value Found (1=yes 0=no)', 'Num Values', 'Efficiency (%)' columns
    df.drop(columns=['Value Found (1=yes 0=no)', 'Efficiency (%)'], errors='ignore', inplace=True)

    # Add a column for the version
    version = format_version_name(os.path.basename(file_path))
    df.insert(0, 'Version', version)

    # Add columns for Speedup and Efficiency
    df['Speedup'] = 1  # Default value for sequential

    # Calculate Speedup and Efficiency if parallel data is provided
    if seq_df is not None and 'Sequential' not in df['Version'].values:
        df['Speedup'] = seq_df['Total Program Time (s)'].astype(float) / df['Total Program Time (s)'].astype(float)
        df['Speedup'] = df['Speedup'].apply(lambda x: f"{x:.7f}")
    else:
        df['Speedup'] = 1
    
    df.rename(columns={'Block Size': 'CUDA Threads per Block'}, inplace=True)

    # Ensure 'Sequential' is the first row
    df = df.sort_values(by='Version', ascending=True if 'Sequential' in df['Version'].values else False)

    return df

def create_performance_tables(base_dirs, output_dir):
    """
    Generates separate performance tables for different optimization levels, 
    values, and version types (CUDA+OMP, MPI+OMP), including sequential version data.
    
    Args:
    - base_dirs (list): List of base directories for sequential, MPI/OpenMP, and CUDA/OpenMP data.
    - output_dir (str): Output directory for generated tables.
    """
    optimization_levels = ['opt0', 'opt1', 'opt2', 'opt3']

    for optimization_level in optimization_levels:
        seq_dir = os.path.join(base_dirs[0], optimization_level)
        mpi_dir = os.path.join(base_dirs[1], optimization_level)
        cuda_dir = os.path.join(base_dirs[2], optimization_level)

        if not os.path.exists(seq_dir):
            continue

        seq_files = {int(file.split('_')[-1].split('.')[0]): file for file in os.listdir(seq_dir) if file.endswith('.csv')}

        for num_values, seq_file in seq_files.items():
            seq_df = read_and_process_csv(os.path.join(seq_dir, seq_file))
            if 'Num Values' in seq_df.columns:
                seq_df.drop(columns=['Num Values'], inplace=True)

            if os.path.exists(mpi_dir):
                mpi_combined_dfs = [seq_df]
                mpi_files = [file for file in os.listdir(mpi_dir) if file.endswith('.csv') and int(file.split('_')[-1].split('.')[0]) == num_values]
                for mpi_file in mpi_files:
                    mpi_df = read_and_process_csv(os.path.join(mpi_dir, mpi_file), seq_df)
                    mpi_combined_dfs.append(mpi_df)

                mpi_final_df = pd.concat(mpi_combined_dfs, ignore_index=True)
                mpi_final_df.sort_values(by=['Num Values', 'OMP Threads'], ascending=[True, True], inplace=True)
                mpi_final_df.insert(0, 'SortOrder', mpi_final_df['Version'].apply(lambda x: 0 if 'Sequential' in x else 1))
                mpi_final_df.sort_values(by=['SortOrder', 'Num Values', 'OMP Threads'], inplace=True)
                mpi_final_df.drop(columns=['SortOrder'], inplace=True)
                generate_table_png(mpi_final_df, os.path.join(output_dir, f'mpi_table_{optimization_level}_{num_values}.png'), optimization_level, num_values)

            if os.path.exists(cuda_dir):
                cuda_combined_dfs = [seq_df]
                cuda_files = [file for file in os.listdir(cuda_dir) if file.endswith('.csv') and int(file.split('_')[-1].split('.')[0]) == num_values]
                for cuda_file in cuda_files:
                    cuda_df = read_and_process_csv(os.path.join(cuda_dir, cuda_file), seq_df)
                    cuda_combined_dfs.append(cuda_df)

                cuda_final_df = pd.concat(cuda_combined_dfs, ignore_index=True)
                cuda_final_df.sort_values(by=['Num Values', 'OMP Threads'], ascending=[True, True], inplace=True)
                cuda_final_df.insert(0, 'SortOrder', cuda_final_df['Version'].apply(lambda x: 0 if 'Sequential' in x else 1))
                cuda_final_df.sort_values(by=['SortOrder', 'Num Values', 'OMP Threads'], inplace=True)
                cuda_final_df.drop(columns=['SortOrder'], inplace=True)
                generate_table_png(cuda_final_df, os.path.join(output_dir, f'cuda_table_{optimization_level}_{num_values}.png'), optimization_level, num_values)

def generate_table_png(dataframe, file_path, optimization_level, num_values):
    """
    Generates a table PNG image from a DataFrame.
    
    Args:
    - dataframe (pandas.DataFrame): DataFrame to be visualized.
    - file_path (str): File path to save the generated image.
    - optimization_level (str): Optimization level for the table.
    - num_values (int): Number of values for the table.
    """
    fig_width = max(12, len(dataframe.columns) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.axis('tight')
    ax.axis('off')


    # Add a title above the table
    title = f"Optimization Level: {optimization_level}, Num Values: {num_values}"
    plt.title(title, fontsize=10, pad=12)

    # Create the table
    table_columns = [col for col in dataframe.columns if col != 'Num Values']
    table_ax = ax.table(cellText=dataframe[table_columns].values, colLabels=table_columns, loc='center')
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(8)
    table_ax.scale(1.3, 1.3)

    # Set the background color for the first row (headers) to a shade of blue
    for key, cell in table_ax.get_celld().items():
        if key[0] == 0:  # Check if the cell is in the first row (header)
            cell.set_facecolor('#4F81BD')  # Change the color as desired
            cell.set_text_props(color='w')  # Change the text color to white for better readability

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    
def plot_speedup_for_mpi_process(dataframe, mpi_procs, optimization_level, output_dir, num_values, seq_speedup):
    """
    Plots the speedup for a specific number of MPI processes.
    
    Args:
    - dataframe (pandas.DataFrame): DataFrame containing performance data.
    - mpi_procs (int): Number of MPI processes.
    - optimization_level (str): Optimization level for the plot.
    - output_dir (str): Output directory for the generated plot.
    - num_values (int): Number of values for the plot.
    - seq_speedup (float): Sequential speedup for reference.
    
    Returns:
    - str: File path of the generated plot.
    """
    df_filtered = dataframe[(dataframe['MPI Processes'] == mpi_procs) & (dataframe['Version'] != 'Sequential')]
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['OMP Threads'], df_filtered['Speedup'], 'o-', label=f'Speedup for {mpi_procs} MPI Procs')

    for i, row in df_filtered.iterrows():
            plt.text(row['OMP Threads'], row['Speedup'], f"{row['Speedup']:.4f}", fontsize=8, verticalalignment='bottom')

    # Ideal speedup line based on sequential version
    omp_threads = df_filtered['OMP Threads'].unique()
    ideal_speedup_line = seq_speedup * omp_threads
    plt.plot(omp_threads, ideal_speedup_line, 'r--', label='Ideal Speedup')

    plt.title(f'Speedup for Optimization Level {optimization_level} with {num_values} Values and {mpi_procs} MPI Processes')
    plt.xlabel('OMP Threads')
    plt.ylabel('Speedup')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f'speedup_optlevel_{optimization_level}_mpi_{mpi_procs}_numvalues_{num_values}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath
    
def plot_speedup_for_cuda(dataframe, block_sizes, optimization_level, output_dir, num_values, seq_speedup):
    """
    Plots the speedup for a specific block size in CUDA version.
    
    Args:
    - dataframe (pandas.DataFrame): DataFrame containing performance data.
    - block_sizes (list): List of CUDA block sizes.
    - optimization_level (str): Optimization level for the plot.
    - output_dir (str): Output directory for the generated plot.
    - num_values (int): Number of values for the plot.
    - seq_speedup (float): Sequential speedup for reference.
    
    Returns:
    - str: File path of the generated plot.
    """
    plt.figure(figsize=(10, 6))

    for block_size in block_sizes:
        df_filtered = dataframe[(dataframe['CUDA Threads per Block'] == block_size) & (dataframe['Version'] != 'Sequential')]
        plt.plot(df_filtered['OMP Threads'], df_filtered['Speedup'], 'o-', label=f'Speedup for Block Size {block_size}')

        for i, row in df_filtered.iterrows():
            plt.text(row['OMP Threads'], row['Speedup'], f"{row['Speedup']:.4f}", fontsize=8, verticalalignment='bottom')

    # Ideal speedup line based on sequential version
    omp_threads = df_filtered['OMP Threads'].unique()
    ideal_speedup_line = seq_speedup * omp_threads
    plt.plot(omp_threads, ideal_speedup_line, 'r--', label='Ideal Speedup')

    plt.title(f'Speedup for Optimization Level {optimization_level} with {num_values} Values and CUDA Version')
    plt.xlabel('OMP Threads')
    plt.ylabel('Speedup')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f'speedup_optlevel_{optimization_level}_cuda_numvalues_{num_values}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def create_performance_plots(base_dirs, plot_output_dir):
    """
    Generates performance plots for different optimization levels and values.
    
    Args:
    - base_dirs (list): List of base directories for sequential and MPI/OpenMP data.
    - plot_output_dir (str): Output directory for generated plots.
    """
    optimization_levels = ['opt0', 'opt1', 'opt2', 'opt3']

    for optimization_level in optimization_levels:
        seq_dir = os.path.join(base_dirs[0], optimization_level)
        mpi_dir = os.path.join(base_dirs[1], optimization_level)
        cuda_dir = os.path.join(base_dirs[2], optimization_level)

        seq_files = {int(file.split('_')[-1].split('.')[0]): file for file in os.listdir(seq_dir) if file.endswith('.csv')}

        for num_values, seq_file in seq_files.items():
            seq_df = read_and_process_csv(os.path.join(seq_dir, seq_file))
            mpi_files = [file for file in os.listdir(mpi_dir) if file.endswith('.csv') and int(file.split('_')[-1].split('.')[0]) == num_values]
            cuda_files = [file for file in os.listdir(cuda_dir) if file.endswith('.csv') and int(file.split('_')[-1].split('.')[0]) == num_values]

            combined_df = pd.DataFrame()
            for mpi_file in mpi_files:
                mpi_df = read_and_process_csv(os.path.join(mpi_dir, mpi_file), seq_df)
                combined_df = pd.concat([combined_df, mpi_df], ignore_index=True)

            cuda_df = pd.DataFrame()
            for cuda_file in cuda_files:
                tmp_df = read_and_process_csv(os.path.join(cuda_dir, cuda_file), seq_df)
                cuda_df = pd.concat([cuda_df, tmp_df], ignore_index=True)

            combined_df['Speedup'] = combined_df['Speedup'].astype(float)
            cuda_df['Speedup'] = cuda_df['Speedup'].astype(float)

            combined_df.sort_values(by=['MPI Processes', 'OMP Threads'], ascending=True, inplace=True)
            cuda_df.sort_values(by=['CUDA Threads per Block', 'OMP Threads'], ascending=True, inplace=True)

            image_paths_mpi = []
            mpi_process_counts = combined_df['MPI Processes'].unique()
            for mpi_procs in mpi_process_counts:
                image_path_mpi = plot_speedup_for_mpi_process(combined_df, mpi_procs, optimization_level, plot_output_dir, num_values, 1)
                image_paths_mpi.append(image_path_mpi)

            # Choose CUDA block sizes for plotting
            block_sizes = cuda_df['CUDA Threads per Block'].unique()[:3]  # Choose the first three block sizes

            image_path_cuda = plot_speedup_for_cuda(cuda_df, block_sizes, optimization_level, plot_output_dir, num_values, 1)
    
# Define base directories and output directories
base_dirs = ['./data/SequentialCSVResult', './data/MPIOpenMPCSVResult', './data/CUDAOpenMPCSVResult']
table_output_dir = './plot_and_tables/PerformanceTable'
plot_output_dir = './plot_and_tables/PerformancePlot'

# Call functions to create performance tables and plots
create_performance_tables(base_dirs, table_output_dir)
create_performance_plots(base_dirs, plot_output_dir)