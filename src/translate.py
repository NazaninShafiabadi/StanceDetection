"""
Languages in FLORES-200:
German:	deu_Latn
French:	fra_Latn
Italian: ita_Latn
English: eng_Latn

Sample usage:

python src/translate.py \
--model="facebook/nllb-200-distilled-600M" \
--dataset="translations/it2fr.csv" \
--split="train" \
--src_lang="fra_Latn" \
--tgt_lang="ita_Latn" \
--batch_size=128 \
--max_len=128 \
--output_file="translations/it2fr2it_test.csv"
"""

import argparse
from datasets import load_dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging
from tqdm import tqdm

logging.set_verbosity_error()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/nllb-200-distilled-600M', type=str, help='Model name or path')
    parser.add_argument('--dataset', required=True, type=str, help='HuggingFace dataset')
    parser.add_argument('--split', required=True, type=str, help='Dataset split')
    parser.add_argument('--src_lang', required=True, type=str, help='Source language')
    parser.add_argument('--tgt_lang', required=True, type=str, help='Target language')
    parser.add_argument('--filter', action='store_true', help='Filter dataset by source language')
    parser.add_argument('--max_len', default=512, type=int, help='Maximum sequence length')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--output_file', default='translations.csv', type=str, help='Output directory')
    return parser


def main(args):
    # Load the model and tokenizer
    print('Loading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model  #, src_lang=args.src_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)

    # Prepare inputs
    print('Preparing inputs...')
    if args.dataset.endswith('.csv'):
        dataset = load_dataset('csv', data_files=args.dataset, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    if args.filter:
        dataset = dataset.filter(lambda x: x['language'] == args.src_lang[:2])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Translate
    questions, comments = [], []
    for batch in tqdm(data_loader, desc="Translating", unit="batch"):
        tokenized_q = tokenizer(batch['question'], padding='longest', truncation=True, 
                                max_length=args.max_len, return_tensors='pt').to(DEVICE)
        tokenized_c = tokenizer(batch['comment'], padding='longest', truncation=True, 
                                max_length=args.max_len, return_tensors='pt').to(DEVICE)
        
        translated_q = model.generate(**tokenized_q, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.tgt_lang))
        translated_c = model.generate(**tokenized_c, forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.tgt_lang))
        
        questions.extend(tokenizer.batch_decode(translated_q, skip_special_tokens=True))
        comments.extend(tokenizer.batch_decode(translated_c, skip_special_tokens=True))
    
    # Replace old values with new translations
    translated_ds = dataset.map(lambda example, idx: {
        "question": questions[idx], 
        "comment": comments[idx]
        }, with_indices=True)
    translated_ds.to_csv(args.output_file, index=False)
    print(f'Translations saved to {args.output_file}')

    return


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)