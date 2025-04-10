import pandas as pd
import numpy as np
import os

class MeanError:
    def __init__(self, groundtruth_path, pose_name, blaze_df):
        """
        Initialize the MeanError calculator with ground truth path and blaze dataframe.
        
        Args:
            groundtruth_path (str): Path to directory containing ground truth CSV files
            pose_name (str): Name of the pose (used to construct filename)
            blaze_df (pd.DataFrame): DataFrame containing blaze pose estimation results
        """
        self.groundtruth_path = groundtruth_path
        self.pose_name = pose_name
        self.blaze_df = blaze_df.copy()  # Use a copy to avoid modifying the input
        self.precision = 4

        # Construct ground truth file path
        self.groundtruth_file = os.path.join(groundtruth_path, f"Dataset_{pose_name}.csv")
        
        try:
            # Read ground truth data and drop first column
            self.groundtruth_df_orig = pd.read_csv(self.groundtruth_file)
            self.groundtruth_df = pd.read_csv(self.groundtruth_file).iloc[:, 1:]
        except Exception as e:
            raise ValueError(f"Failed to load ground truth data: {str(e)}")

        # Validate the DataFrames (ignore column count mismatch)
        self._validate_dataframe_header()
        
        # Columns to compare (all columns since we dropped the first one)
        self.columns_to_compare = self.groundtruth_df.columns
    
    def _validate_dataframe_header(self):
        # """Validate that the DataFrames have compatible shapes (ignoring column count)."""
        # if self.groundtruth_df.shape[0] != self.blaze_df.shape[0]:
        #     raise ValueError("The DataFrames must have the same number of rows.")
        
        # # Only check that blaze_df has at least as many columns as groundtruth_df
        # if self.groundtruth_df.shape[1] > self.blaze_df.shape[1]:
        #     raise ValueError("Blaze DataFrame must have at least as many columns as ground truth (minus first column).")
        
        # Verify column names match for the columns that exist in both
        common_cols = min(len(self.groundtruth_df.columns), len(self.blaze_df.columns))
        if not all(self.groundtruth_df.columns[:common_cols] == self.blaze_df.columns[:common_cols]):
            raise ValueError("Column headers must match for the comparable columns.")

    # [Rest of your methods remain the same, but will now work with the aligned columns]
    
    def calculate_mean_error(self):   
        """
        Calculate the overall mean absolute error between comparable values.
        """
        # Get numeric columns from both DataFrames (excluding first column if needed)
        gt_numeric = self.groundtruth_df.select_dtypes(include=[np.number])
        blaze_numeric = self.blaze_df.select_dtypes(include=[np.number])

        error = np.abs(gt_numeric - blaze_numeric)
        #print(f"Error: {error}")
        mean_error = error.values.mean()
        #print(f"Mean Error shape: {mean_error.shape}")
        return mean_error
    
    def calculate_columnwise_mean_error(self):   
        """
        Calculate the mean absolute error for each column.
        
        Returns:
            dict: Dictionary of {column_name: mean_error}
        """
        errors = {}
        for column in self.columns_to_compare:
            col_error = np.abs(self.df1[column] - self.df2[column])
            errors[column] = col_error.mean()
        return errors
    
    def calculate_mean_error_by_suffix(self):
        """
        Calculate the mean error grouped by coordinate suffix (x, y, z) and visibility (s).
        
        Returns:
            tuple: (x_mean_error, y_mean_error, z_mean_error, visibility_mean_error)
        """
        x_error = y_error = z_error = vis_error = 0.0
        x_count = y_count = z_count = vis_count = 0
        
        for column in self.columns_to_compare:
            if column[-1] == "x":
                col_error = np.abs(self.groundtruth_df[column] - self.blaze_df[column])
                x_error += col_error.sum()
                x_count += len(col_error)
            elif column[-1] == "y":
                col_error = np.abs(self.groundtruth_df[column] - self.blaze_df[column])
                y_error += col_error.sum()
                y_count += len(col_error)
            elif column[-1] == "z":
                col_error = np.abs(self.groundtruth_df[column] - self.blaze_df[column])
                z_error += col_error.sum()
                z_count += len(col_error)
            elif column[-1] == "s":
                col_error = np.abs(self.groundtruth_df[column] - self.blaze_df[column])
                vis_error += col_error.sum()
                vis_count += len(col_error)
        
        # Calculate means (avoid division by zero)
        x_mean = x_error / x_count if x_count > 0 else 0
        y_mean = y_error / y_count if y_count > 0 else 0
        z_mean = z_error / z_count if z_count > 0 else 0
        vis_mean = vis_error / vis_count if vis_count > 0 else 0
        
        return x_mean, y_mean, z_mean, vis_mean
    
    def calculate_mean_error_per_row(self):
        """
        Calculate the mean absolute error for each row (excluding the first column).
        
        Returns:
            dict: Dictionary of {row_index: mean_error}
        """
        numeric_cols = self.groundtruth_df.select_dtypes(include=[np.number]).columns[1:]
        abs_errors = np.abs(self.groundtruth_df[numeric_cols] - self.blaze_df[numeric_cols])
        mean_errors_per_row = abs_errors.mean(axis=1)
        return mean_errors_per_row.to_dict()
    
    def calculate_mean_error_per_suffix_per_row(self, row):
        """
        Calculate the mean absolute error per suffix (x, y, z, s) for a specific row.
        
        Args:
            row (int): Row index to calculate errors for
            
        Returns:
            tuple: (x_mean, y_mean, z_mean, vis_mean, total_error)
        """
        x_error = y_error = z_error = vis_error = 0.0
        x_count = y_count = z_count = vis_count = 0
        
        for column in self.columns_to_compare:
            if column[-1] == "x":
                col_error = np.abs(self.groundtruth_df[column][row] - self.blaze_df[column][row])
                x_error += col_error
                x_count += 1
            elif column[-1] == "y":
                col_error = np.abs(self.groundtruth_df[column][row] - self.blaze_df[column][row])
                y_error += col_error
                y_count += 1
            elif column[-1] == "z":
                col_error = np.abs(self.groundtruth_df[column][row] - self.blaze_df[column][row])
                z_error += col_error
                z_count += 1
            elif column[-1] == "s":
                col_error = np.abs(self.groundtruth_df[column][row] - self.blaze_df[column][row])
                vis_error += col_error
                vis_count += 1
        
        
        # Calculate means (avoid division by zero)
        x_mean = x_error / x_count if x_count > 0 else 0
        y_mean = y_error / y_count if y_count > 0 else 0
        z_mean = z_error / z_count if z_count > 0 else 0
        vis_mean = vis_error / vis_count if vis_count > 0 else 0
        
        adjusted_total_error = (x_mean + y_mean + z_mean) / 3  # don't include visibility error in total error
        total_error = (x_mean  + y_mean + z_mean + vis_mean)/4  # don't include visibility error in total error
        
        return x_mean, y_mean, z_mean, vis_mean, adjusted_total_error, total_error
    
    
    def print_all_errors(self):
        """Print all error metrics with the specified precision."""
        try:
            print(f"Mean Error between the two DataFrames: {self.calculate_mean_error():.{self.precision}f}")
            
            suffix_errors = self.calculate_mean_error_by_suffix()
            print(f"\nMean Error by Coordinate Type:")
            print(f"X: {suffix_errors[0]:.{self.precision}f}")
            print(f"Y: {suffix_errors[1]:.{self.precision}f}")
            print(f"Z: {suffix_errors[2]:.{self.precision}f}")
            print(f"Visibility: {suffix_errors[3]:.{self.precision}f}")
            
            print("\nMean Error Per Row (Excluding First Column):")
            row_errors = self.calculate_mean_error_per_row()
            for row_idx, error in row_errors.items():
                print(f"Row {row_idx}: {error:.{self.precision}f}")
                    
        except Exception as e:
            print(f"An error occurred: {e}")