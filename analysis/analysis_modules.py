import os, glob
from typing import List
from scipy.stats import norm
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_accuracy(preds_df, gold_column, pred_column):
    return preds_df.groupby('language')[[gold_column, pred_column]]\
        .apply(lambda x: (x[gold_column] == x[pred_column])\
               .mean())\
                .reset_index()\
                    .rename(columns={0: 'accuracy'})


def plot(df):
    topics = df.index
    languages = df.columns
    x = np.arange(len(topics))  # label locations
    width = 0.2  # bar width

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'de': 'indianred', 'fr': 'royalblue', 'it': 'seagreen'}

    # Plot bars for each language
    for i, lang in enumerate(languages):
        ax.bar(x + i * width, df[lang], width, color=colors[lang], label=lang)

    ax.set_xlabel("Topic")
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance by Topic and Language")
    ax.set_xticks(x + width, topics, rotation=45, ha="right")
    ax.legend(loc='upper center', ncol=len(languages))
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


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