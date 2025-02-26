"""
Sample usage:

python src/predict.py \
--model='models/binary_stance_classifier/best_model' \
--test_file="data/xstance/test.jsonl" \
--output_file="predictions/bi_test_preds.csv" \
--batch_size=128 \
--label_mapping_file='models/binary_stance_classifier/label_mapping.json' \
--is_finetuned
"""

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
from preprocessing import read_data_to_df, InputPreprocessor

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='finetuned_model', type=str, help='Model name or path')
    parser.add_argument('--test_file', required=True, type=str, help='Path to test data')
    parser.add_argument('--output_file', default='predictions.csv', type=str, help='File to save predictions')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--max_len', default=512, type=int, help='Maximum sequence length')
    parser.add_argument('--ignore_questions', action='store_true', help='Ignore questions during tokenization')
    parser.add_argument('--ignore_comments', action='store_true', help='Ignore comments during tokenization')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels')
    parser.add_argument('--label_mapping_file', default=None, type=str, help='Path to label mapping file')
    parser.add_argument('--is_finetuned', action='store_true', help='Flag for using fine-tuned model')
    return parser


# Custom classification head for vanilla models (not fine-tuned)
class StanceDetectionHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(StanceDetectionHead, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)  # Classifier head
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear(x)
        return self.softmax(logits)


def predict(args):  
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Check if model is fine-tuned or vanilla (pre-trained)
    if args.is_finetuned:
        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE)
    else:
        # Load the base model (vanilla, not fine-tuned)
        base_model = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE)
        model = StanceDetectionHead(input_dim=base_model.config.hidden_size, num_labels=args.num_labels).to(DEVICE)

    input_preprocessor = InputPreprocessor(tokenizer, 
                                           max_len=args.max_len, 
                                           ignore_questions=args.ignore_questions, 
                                           ignore_comments=args.ignore_comments, 
                                           device=DEVICE)
    
    if args.label_mapping_file:
        # Load label mapping
        input_preprocessor.load_label_mapping(args.label_mapping_file)
    
    dataset = input_preprocessor.process(args.test_file)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Make predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting", unit="batch"):
            if args.is_finetuned:
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            else:
                # For vanilla model, use the custom classification head
                outputs = base_model(**batch, output_hidden_states=True)
                final_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, seq_length, hidden_dim]
                cls_token = final_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
                probs = model(cls_token)
                preds = probs.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds)
    
    # Convert predictions back to original label names
    label_mapping = input_preprocessor.label_mapping  # {label_name: index}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predictions = [reverse_label_mapping[pred] for pred in predictions]
    
    # Save predictions to output file
    df = read_data_to_df(args.test_file)
    df['prediction'] = predictions
    df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    predict(args)