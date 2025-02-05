""" 
Sample usage:

python src/finetune.py \
--model="xlm-roberta-base" \
--train_file="data/xstance/train.jsonl" \
--val_file="data/xstance/valid.jsonl" \
--output_dir="models/binary_stance_classifier" \
--balance_by language label \
--max_len=128 \
--num_epochs=10 \
--batch_size=128
--ignore_questions

python src/finetune.py \
--model="xlm-roberta-base" \
--train_file="data/xstance/train.jsonl" \
--val_file="data/xstance/valid.jsonl" \
--output_dir="models/multi_macro_stance_classifier" \
--balance_by language label \
--max_len=256 \
--label_column='numerical_label' \
--num_labels=4 \
--num_epochs=10 \
--batch_size=128 \
--average='macro'
"""


import os
import argparse
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
from preprocessing import InputPreprocessor

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
    parser.add_argument('--max_len', default=512, type=int, help='Maximum sequence length')
    parser.add_argument('--ignore_questions', action='store_true', help='Ignore questions during tokenization')
    parser.add_argument('--ignore_comments', action='store_true', help='Ignore comments during tokenization')
    parser.add_argument('--label_column', default='label', type=str, help='Label column name')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--average', default='binary', type=str, help='Average method for metrics. Choose from: binary, micro, macro, weighted')
    return parser


# def compute_metrics(pred):
#     logits, labels = pred
#     predictions = logits.argmax(axis=-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
#     acc = accuracy_score(labels, predictions)
#     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics(average_method):
    def _compute_metrics(pred):
        logits, labels = pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average_method)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    return _compute_metrics


def main(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels).to(DEVICE)

    input_preprocessor = InputPreprocessor(tokenizer, 
                                           max_len=args.max_len, 
                                           ignore_questions=args.ignore_questions, 
                                           ignore_comments=args.ignore_comments,
                                           device=DEVICE
                                           )
    trainset = input_preprocessor.process(args.train_file, label_column=args.label_column, balance_by=args.balance_by)
    validset = input_preprocessor.process(args.val_file, label_column=args.label_column)
    
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir=os.path.join(args.output_dir, 'logs'),
    learning_rate=2e-5,
    warmup_ratio=0.1,  # Use 10% of training steps for warm-up
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # fp16=True if DEVICE=='cuda' else False  # Enable mixed precision
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=validset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(args.average),
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

    # Save label mapping for consistency during inference
    input_preprocessor.save_label_mapping(os.path.join(args.output_dir, 'label_mapping.json'))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)