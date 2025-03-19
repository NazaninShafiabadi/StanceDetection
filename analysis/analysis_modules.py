import os, glob
from typing import List, Union
from scipy.stats import norm
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_accuracy(df, gold_column, pred_column=None, group_by:Union[List, str]='language', baseline=False):
    if baseline:
        MFC = df[gold_column].mode()[0]
        accuracy = df.groupby(group_by)[gold_column].apply(lambda x: (x == MFC).mean())
    else: 
        if not pred_column:
            raise ValueError("pred_column must be specified when baseline is False.")
        accuracy = df.groupby(group_by)[[gold_column, pred_column]].apply(
            lambda x: (x[gold_column] == x[pred_column]).mean()
        )
    return accuracy.reset_index(name='accuracy')


def add_accuracy_diff(acc_df, baseline_col: str, diff_mode='baseline'):
    """
    Adds accuracy difference annotations to a DataFrame.
    
    baseline_col: either the first numerical column (in 'previous' mode) or the column 
    against which to compare all other columns (in 'baseline' mode)
    diff_mode: either 'baseline' or 'previous'
    """
    # Validate diff_mode
    if diff_mode not in ['baseline', 'previous']:
        raise ValueError("diff_mode must be either 'baseline' or 'previous'.")

    # Check if the baseline column exists
    if baseline_col not in acc_df.columns:
        raise ValueError(f"Column '{baseline_col}' not found in the DataFrame.")

    # Check if the baseline column is numeric (if using 'baseline' mode)
    if diff_mode == 'baseline' and not pd.api.types.is_numeric_dtype(acc_df[baseline_col]):
        raise ValueError(f"Baseline column '{baseline_col}' must be numeric.")

    # Check if the DataFrame has at least two numerical columns for 'previous' mode
    num_cols = acc_df.select_dtypes(include='number').columns
    if diff_mode == 'previous' and len(num_cols) < 2:
        raise ValueError("At least two numerical columns are required for 'previous' mode.")

    df = acc_df.copy()
    df[num_cols] = (df[num_cols] * 100).round(2)

    # Create a new DataFrame to store formatted values
    formatted_df = df.copy()

    for i, col in enumerate(num_cols):
        if col == baseline_col:
            continue

        ref_col = baseline_col if diff_mode == 'baseline' else num_cols[i - 1]
        formatted_df[col] = df.apply(lambda row: f"{row[col]:.2f}% ({row[col] - row[ref_col]:+.2f}%)", axis=1)

    # Format the baseline column
    formatted_df[baseline_col] = df[baseline_col].apply(lambda x: f"{x}%")
    
    return formatted_df


def plot_metrics(model_dir):
    ckpt_files = glob.glob(os.path.join(model_dir, 'checkpoint-*'))
    
    # Sort the files by modification time
    ckpt_files.sort(key=os.path.getmtime)

    if ckpt_files:
        last_ckpt = ckpt_files[-1]  
    else: 
        return f"No checkpoint files in {model_dir}"

    trainer_state = os.path.join(last_ckpt, 'trainer_state.json')

    with open(trainer_state, 'r') as file:
        data = json.load(file)

    log_history = (pd.DataFrame(data['log_history'])
                .groupby("epoch")
                .agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)
                .reset_index())
    
    log_history.plot(x='epoch', 
                     y=['eval_accuracy', 'eval_loss', 'loss'], 
                     label=['Validation Accuracy', 'Validation Loss', 'Training Loss'], 
                     color=['green', 'red', 'maroon'], 
                     grid=True
                     )


def wald_ci(p, n, confidence=0.95):
    """
    Computes a confidence interval using the Gaussian (Wald) method.

    Args:
        p (float): Proportion of errors (or any rate being measured).
        n (int): Sample size.
        confidence (float): Confidence level (default is 0.95 for 95% CI).

    Returns:
        str: Confidence interval as a formatted string "[lower, upper]".
    """
    if n == 0:
        return "[0, 0]"
    
    z = norm.ppf(1 - (1 - confidence) / 2)  # Compute z-score dynamically
    se = np.sqrt(p * (1 - p) / n)  # Standard error
    lower = max(0, p - z * se)  # Clamp at 0 to avoid negative proportions
    upper = min(1, p + z * se)  # Clamp at 1 to avoid >100% values
    
    return f"[{round(lower * 100, 2)}, {round(upper * 100, 2)}]"

def analyze_errors(df, gold_column, pred_column, group_cols, confidence=0.95, compute_ci=False):
    """
    Analyzes prediction errors by language, computing error rates and (optional) confidence intervals.
    """
    def compute_stats(group):
        n = len(group)
        err_rate = np.mean(group[gold_column] != group[pred_column])
        fp_rate = np.mean((group[gold_column] == 'AGAINST') & (group[pred_column] == 'FAVOR'))
        fn_rate = np.mean((group[gold_column] == 'FAVOR') & (group[pred_column] == 'AGAINST'))

        result = {
            'ErrRate (%)': round(err_rate * 100, 2),
            'FP (%)': round(fp_rate * 100, 2),
            'FN (%)': round(fn_rate * 100, 2)
        }

        if compute_ci:
            result.update({
                'ErrRate_CI': wald_ci(err_rate, n, confidence),
                'FP_CI': wald_ci(fp_rate, n, confidence),
                'FN_CI': wald_ci(fn_rate, n, confidence)
            })
        
        return pd.Series(result)
    
    return df.groupby(group_cols)[[gold_column, pred_column]].apply(compute_stats).reset_index()


def balance_df(df, columns:List, rs=42):
    # minimum count of rows for any combination of values in the specified columns
    min_count = df.groupby(columns).size().min()
    
    # sample min_count rows for each column combination
    balanced_df = (
        df.groupby(columns)
          .apply(lambda group: group.sample(min_count, random_state=rs))
          .reset_index(drop=True)
    )
    
    return balanced_df.sample(frac=1).reset_index(drop=True)    # shuffle the rows