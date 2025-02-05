"""
Languages in FLORES-200:
German:	deu_Latn
French:	fra_Latn
Italian: ita_Latn
English: eng_Latn

Sample usage:

python src/translate.py \
--model="facebook/nllb-200-distilled-600M" \
--dataset="ZurichNLP/x_stance" \
--split="test" \
--src_lang="ita_Latn" \
--tgt_lang="fra_Latn" \
--batch_size=128 \
--max_len=256 \
--output_file="translations/it2fr.csv"
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
    parser.add_argument('--max_len', default=512, type=int, help='Maximum sequence length')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--output_file', default='translations.csv', type=str, help='Output directory')
    return parser


def main(args):
    # Load the model and tokenizer
    print('Loading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, src_lang=args.src_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)

    # Prepare inputs
    print('Preparing inputs...')
    dataset = load_dataset(args.dataset, split=args.split)
    filtered_dataset = dataset.filter(lambda x: x['language'] == args.src_lang[:2])
    data_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=args.batch_size)

    sep_token = tokenizer.sep_token
    
    # Translate
    translations = []
    for batch in tqdm(data_loader, desc="Translating", unit="batch"):
        input_texts = [q + f' {sep_token} ' + c for q, c in zip(batch["question"], batch["comment"])]
        inputs = tokenizer(input_texts, padding='longest', truncation=True, max_length=args.max_len, return_tensors='pt').to(DEVICE)
        outputs = model.generate(**inputs, 
                                 forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.tgt_lang))
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    
    # Save translations 
    filtered_dataset = filtered_dataset.add_column('translation', translations)
    filtered_dataset.to_csv(args.output_file, index=False)
    print(f'Translations saved to {args.output_file}')

    return


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)