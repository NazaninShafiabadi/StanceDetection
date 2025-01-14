"""
Languages in FLORES-200:
German:	deu_Latn
French:	fra_Latn
Italian: ita_Latn
English: eng_Latn

Sample usage:

python src/translate.py \
--model="facebook/nllb-200-distilled-600M" \
--article="data/xstance/test_it.jsonl" \
--src_lang="ita_Latn" \
--tgt_lang="fra_Latn" \
--batch_size=128 \
--output_file="translations/it2fr.csv"
"""

import pandas as pd
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging
from tqdm import tqdm
from preprocessing import read_data, preprocess, tokenize

logging.set_verbosity_error()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/nllb-200-distilled-600M', type=str, help='Model name or path')
    parser.add_argument('--article', required=True, type=str, help='Article to translate')
    parser.add_argument('--src_lang', required=True, type=str, help='Source language')
    parser.add_argument('--tgt_lang', required=True, type=str, help='Target language')
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

    # Prepare the input data
    article = read_data(args.article)
    preprocessed = preprocess(article)
    sentences = preprocessed['comment'].to_list()
    
    translations = []
    for i in tqdm(range(0, len(sentences), args.batch_size), desc="Translating"):
        batch_sentences = sentences[i : i + args.batch_size]
        inputs = tokenize(batch_sentences, tokenizer, device=DEVICE)

        # Generate translations for the batch
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.tgt_lang),
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

        # Decode translations
        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)

    # Save translations to file
    output = pd.concat([article.drop(columns=['comment']), pd.Series(translations, name='comment')], axis=1)
    output.to_csv(args.output_file, index=False)
    print(f'Translations saved to {args.output_file}')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)