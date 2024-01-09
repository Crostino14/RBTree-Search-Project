"""
Course: High Performance Computing 2023/2024

Lecturer: Francesco Moscato      fmoscato@unisa.it

Student and Creator:
Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it

Source Code for creating tables of parallel (MPI+OMP, CUDA+OMP) and sequential version's data.

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
import numpy as np

def format_float(value):
    """
    Formats a floating-point number with nine decimal places.

    Args:
    value (float): The floating-point number to be formatted.

    Returns:
    str: Formatted number with nine decimal places.
    """
    return f"{value:.9f}"


def format_version_name(version):
    """
    Formats the version name by removing the "csv_" prefix and replacing underscores with spaces.

    Args:
    version (str): The version name to be formatted.

    Returns:
    str: Formatted version name.
    """
    return version.replace("csv_", "").replace("_", " ")

def read_and_create_tables(csv_dirs):
    """
    Reads CSV files from specified directories and creates formatted tables.

    Args:
    csv_dirs (list): A list of directory paths containing CSV files.

    Returns:
    None
    """
    for csv_dir in csv_dirs:
        optimization_level = csv_dirs.index(csv_dir) % 4
        version_name = csv_dir.split('/')[2].replace('CSVResult', '')

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        tables = []

        for file in csv_files:
            file_path = os.path.join(csv_dir, file)
            df = pd.read_csv(file_path)
            df['Version'] = format_version_name(os.path.splitext(file)[0])

            # Convert columns to the correct types
            df['OMP Threads'] = df['OMP Threads'].astype(int)
            df['MPI Processes'] = df['MPI Processes'].astype(int)
            df['Block Size'] = df['Block Size'].astype(int)
            df['Num Values'] = df['Num Values'].astype(int)
            df['Search Time (s)'] = df['Search Time (s)'].apply(format_float)
            df['Total Program Time (s)'] = df['Total Program Time (s)'].apply(format_float)

            if 'CUDA' in version_name:
                df.rename(columns={'Search Time (s)': 'GPU Search Time (s)'}, inplace=True)
                df['CPU Time (s)'] = df['Total Program Time (s)'].astype(float) - df['GPU Search Time (s)'].astype(float)
                df['CPU Time (s)'] = df['CPU Time (s)'].apply(format_float)
                
                # Define the columns and their order for the final table (including CPU Time)
                final_table_cols = ['Version', 'OMP Threads', 'MPI Processes', 'Num Values',
                                    'GPU Search Time (s)', 'CPU Time (s)', 'Total Program Time (s)']
            else:
                final_table_cols = ['Version', 'OMP Threads', 'MPI Processes', 'Num Values',
                                    'Search Time (s)', 'Total Program Time (s)']

            df = df[final_table_cols]
            tables.append(df)

        if tables:
            final_table = pd.concat(tables)
            output_dir = os.path.join('./plot_and_tables', version_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Generating the table image
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')

            table_data = [list(final_table.columns)] + final_table.values.tolist()

            table_ax = ax.table(cellText=table_data, loc='center', colWidths=[0.15]*len(final_table.columns))
            table_ax.auto_set_font_size(False)
            table_ax.set_fontsize(8)
            table_ax.scale(1.2, 1.2)

            # Set the background color for the first row (headers) to a shade of blue
            for key, cell in table_ax.get_celld().items():
                if key[0] == 0:  # Check if the cell is in the first row (header)
                    cell.set_facecolor('#4F81BD')  # Change the color as desired
                    cell.set_text_props(color='w')  # Change the text color to white for better readability

            plt.savefig(os.path.join(output_dir, f'table_opt{optimization_level}.png'), bbox_inches='tight', pad_inches=0.05)

csv_dirs = ['./data/SequentialCSVResult/opt0', './data/SequentialCSVResult/opt1', 
            './data/SequentialCSVResult/opt2', './data/SequentialCSVResult/opt3',
            './data/CUDAOpenMPCSVResult/opt0', './data/CUDAOpenMPCSVResult/opt1',
            './data/CUDAOpenMPCSVResult/opt2', './data/CUDAOpenMPCSVResult/opt3',
            './data/MPIOpenMPCSVResult/opt0', './data/MPIOpenMPCSVResult/opt1',
            './data/MPIOpenMPCSVResult/opt2', './data/MPIOpenMPCSVResult/opt3']

read_and_create_tables(csv_dirs)