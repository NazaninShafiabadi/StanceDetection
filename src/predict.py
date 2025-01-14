""" 
Sample usage:

python src/predict.py \
--model='xlm-roberta-base' \
--test_file="data/xstance/test.jsonl" \
--output_file="predictions/xstance_test_preds.csv" \
--batch_size=128
"""

import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time
import torch
from preprocessing import read_data, preprocess

DEVICE = 0 if torch.cuda.is_available() else -1
print(f'Using device: {DEVICE}')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='finetuned_model', type=str, help='Model name or path')
    parser.add_argument('--test_file', required=True, type=str, help='Path to test data')
    parser.add_argument('--output_file', default='predictions.csv', type=str, help='File to save predictions')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    return parser


def predict(df, model):
    inputs = preprocess(df, columns_to_keep=['language', 'comment', 'label', 'numerical_label', 'topic'])
    predictions = model(inputs['comment'].to_list())

    key_mapping = {'label': 'prediction', 'score': 'confidence'}

    predictions = pd.DataFrame(predictions).rename(columns=key_mapping)
    predictions['prediction'] = predictions['prediction'].str.extract(r'(\d+)$').astype(int)

    return pd.concat([inputs, predictions], axis=1)


def main(args):
    # Load the finetuned model
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, batch_size=args.batch_size, device=DEVICE)

    # Predict on test data
    test = read_data(args.test_file)
    print('Inference in progress...')
    start_time = time.time()
    preds = predict(test, classifier)
    print(f"Inference completed in {time.time() - start_time:.2f} seconds.")

    # Save predictions
    preds.to_csv(args.output_file, index=False)
    print(f'Predictions saved to {args.output_file}')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)