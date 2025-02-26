from datasets import Dataset
import pandas as pd
from typing import List, Optional, Union
# import torch
# import zipfile
import pickle


def read_data_to_df(file_path):
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


# def read_zip_file_as_df(zip_file_path):
#     """Reads files from a zip archive into a list of pandas DataFrames."""
#     dfs = []
#     file_readers = {
#         '.csv': pd.read_csv,
#         '.json': pd.read_json,
#         '.jsonl': lambda f: pd.read_json(f, lines=True),
#         '.xlsx': pd.read_excel,
#     }

#     try:
#         with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#             for file_info in zip_ref.infolist():
#                 # print(f"File Name: {file_info.filename}")
#                 ext = file_info.filename.split('.')[-1].lower()
#                 reader = file_readers.get(f'.{ext}')

#                 if reader:
#                     with zip_ref.open(file_info) as file:
#                         dfs.append(reader(file))
#                 else:
#                     print(f"Unsupported file format: {file_info.filename}")

#     except FileNotFoundError:
#         print(f"Error: Zip file not found at {zip_file_path}")
#     except zipfile.BadZipFile:
#         print(f"Error: Invalid zip file at {zip_file_path}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     return dfs


class InputPreprocessor:
    def __init__(self, tokenizer, max_len: int = 512, device='cpu'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.label_mapping = None # storing the label mapping for consistency across train and test data

    def balance_df(self, df, columns:List, rs=42):
        # minimum count of rows for any combination of values in the specified columns
        min_count = df.groupby(columns).size().min()
        
        # sample min_count rows for each column combination
        balanced_df = (
            df.groupby(columns)
            .apply(lambda group: group.sample(min_count, random_state=rs))
            .reset_index(drop=True)
        )
        
        return balanced_df.sample(frac=1).reset_index(drop=True)    # shuffle the rows
    
    def _process_inputs(self, targets: pd.Series, comments: pd.Series) -> List[int]:
        target_tokens = self.tokenizer.encode(targets, add_special_tokens=False)
        comment_tokens = self.tokenizer.encode(comments, add_special_tokens=False)
        sep_token = [self.tokenizer.sep_token_id] if target_tokens and comment_tokens else []
        
        # # truncate the question tokens if the combined length exceeds max_len
        # if len(target_tokens + sep_token + comment_tokens) > self.max_len:
        #     tokens = target_tokens[:self.max_len - len(sep_token + comment_tokens)] + sep_token + comment_tokens
        # else: 
        #     tokens = target_tokens + sep_token + comment_tokens
        tokens = target_tokens + sep_token + comment_tokens
        tokens = tokens[:self.max_len]
        
        return tokens
    
    def _process_labels(self, labels: pd.Series) -> List[int]:
        if self.label_mapping is None:
            unique_labels = labels.unique()
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return labels.map(self.label_mapping).tolist()
    
    def process(self, file, target_col:str, comment_col:str, label_col: Union[str, None] = None, balance_by: Optional[List[str]] = None) -> Dataset:
        df = read_data_to_df(file)

        if balance_by:
            df = self.balance_df(df, balance_by)
        
        token_list = df.apply(lambda row: self._process_inputs(row[target_col], row[comment_col]), axis=1).tolist()
        padded_tokens = self.tokenizer.pad({'input_ids': token_list}, padding='longest')
        
        labels = self._process_labels(df[label_col]) if label_col else None

        dataset = Dataset.from_dict({ 
            "input_ids": padded_tokens["input_ids"], 
            "attention_mask": padded_tokens["attention_mask"], 
            **({"labels": labels} if labels else {}) # only adds a labels key if labels is not None
            })
        dataset.set_format(type='torch', device=self.device)
        
        return dataset

    def save_label_mapping(self, filepath: str):
        with open(filepath, 'wb') as f:  
            pickle.dump(self.label_mapping, f)

    def load_label_mapping(self, filepath: str):
        with open(filepath, 'rb') as f:  
            self.label_mapping = pickle.load(f)