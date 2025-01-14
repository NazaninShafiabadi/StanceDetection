import pandas as pd
from typing import List
import torch
import zipfile


def read_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else:
        print(f"Unsupported file format: {file_path}")
        return
    return df


def read_zip_file_as_df(zip_file_path):
    """Reads files from a zip archive into a list of pandas DataFrames."""
    dfs = []
    file_readers = {
        '.csv': pd.read_csv,
        '.json': pd.read_json,
        '.jsonl': lambda f: pd.read_json(f, lines=True),
        '.xlsx': pd.read_excel,
    }

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # print(f"File Name: {file_info.filename}")
                ext = file_info.filename.split('.')[-1].lower()
                reader = file_readers.get(f'.{ext}')

                if reader:
                    with zip_ref.open(file_info) as file:
                        dfs.append(reader(file))
                else:
                    print(f"Unsupported file format: {file_info.filename}")

    except FileNotFoundError:
        print(f"Error: Zip file not found at {zip_file_path}")
    except zipfile.BadZipFile:
        print(f"Error: Invalid zip file at {zip_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return dfs


def balance_data(df, columns:List, rs=42):
    # minimum count of rows for any combination of values in the specified columns
    min_count = df.groupby(columns).size().min()
    
    # sample min_count rows for each column combination
    balanced_df = (
        df.groupby(columns)
          .apply(lambda group: group.sample(min_count, random_state=rs))
          .reset_index(drop=True)
    )
    
    return balanced_df


def preprocess(df, columns_to_keep=['comment', 'label'], encode_label=True):
    preprocessed_df = df[columns_to_keep]
    if encode_label:
        preprocessed_df.loc[:, 'label'] = preprocessed_df['label'].map({'FAVOR': 1, 'AGAINST': 0})
    return preprocessed_df


def tokenize(sentences:List, tokenizer, labels=None, device='cpu'):
    inputs = tokenizer(sentences, 
                       truncation=True,
                       padding=True,
                       max_length=128,
                       return_tensors="pt").to(device)
    if labels:
        inputs['labels'] = torch.tensor(labels).to(device)
    return inputs