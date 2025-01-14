""" 
Sample usage:

python src/finetune.py \
--model="xlm-roberta-base" \
--train_file="data/xstance/train.jsonl" \
--val_file="data/xstance/valid.jsonl" \
--output_dir="stance_classifier" \
--balance_by language label \
--num_epochs=10 \
--batch_size=128
"""


import os
import argparse
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil
import torch
import warnings
from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification, \
    TrainingArguments, \
    Trainer, \
    logging, \
    EarlyStoppingCallback
from preprocessing import read_data, balance_data, preprocess, tokenize

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Unbabel/xlm-roberta-comet-small', type=str, help='Model name or path')
    parser.add_argument('--train_file', required=True, type=str, help='Path to training data')
    parser.add_argument('--val_file', required=True, type=str, help='Path to validation data')
    parser.add_argument('--output_dir', default='finetuned_model', type=str, help='Output directory')
    parser.add_argument('--balance_by', default=None, nargs='+', help='List of attributes (column names) to balance by. Separate by space.')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    return parser


def compute_metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main(args):
    # Read data
    train_df = read_data(args.train_file)
    valid_df = read_data(args.val_file)
        
    # Balance data if needed
    if args.balance_by:
        train_df = balance_data(train_df, args.balance_by)
        valid_df = balance_data(valid_df, args.balance_by)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels).to(DEVICE)

    # Prepare data
    prep_train = preprocess(train_df)
    train_cms = prep_train['comment'].tolist()
    train_labels = prep_train['label'].to_list()
    train = tokenize(train_cms, tokenizer, train_labels, DEVICE)
    train_dataset = Dataset.from_dict(train)

    prep_valid = preprocess(valid_df)
    val_cms = prep_valid['comment'].to_list()
    val_labels = prep_valid['label'].to_list()
    valid = tokenize(val_cms, tokenizer, val_labels, DEVICE)
    valid_dataset = Dataset.from_dict(valid)
    
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir=os.path.join(args.output_dir, 'logs'),
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    # Training & evaluation
    print("Training...")
    trainer.train()
    results = trainer.evaluate()
    print(results)

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"Best checkpoint selected: {best_ckpt}")

    best_model_dir = os.path.join(args.output_dir, 'best_model')
    trainer.save_model(best_model_dir)

    # Copy the trainer_state from the best checkpoint to the best model directory
    shutil.copy(os.path.join(best_ckpt, 'trainer_state.json'), os.path.join(best_model_dir, 'trainer_state.json'))

    print(f"Best model saved to {best_model_dir}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)