import torch
import argparse
import re
import os
import copy
import numpy as np
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
# from transformers import DistilBertTokenizer, DistilBertForMaskedLM
# from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(modeldir):
    if "bart" not in modeldir:
        tokenizer = AutoTokenizer.from_pretrained(modeldir)
        # Load pre-trained model (weights)
        model = AutoModelForMaskedLM.from_pretrained(modeldir).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{modeldir}")
        # Load pre-trained model (weights)
        model = AutoModelForMaskedLM.from_pretrained(f"facebook/{modeldir}").to(device)
    model.eval()
    return model,tokenizer


def prep_input(input_sents, tokenizer,bert=True):
    for sent in input_sents:
        masked_index = None
        text = []
        mtok = tokenizer.mask_token
        if not bert:
            sent = re.sub('\[MASK\]','X',sent)
            mtok = 'x</w>'
        """
        if "bart" in tokenizer.name_or_path:
            text.append('<s>')
        elif bert: text.append('[CLS]')
        """
        text += sent.strip().split()
        if "bart" in tokenizer.name_or_path:
            text.append(' .')
        elif text[-1] != '.': text.append(' .')
        # if "bart" in tokenizer.name_or_path:
        #     text.append('</s>')
        # elif bert: text.append('[SEP]')
        text = ' '.join(text)
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids[0]
        tokenized_text = [tokenizer.decode([id]) for id in ids]
        for i,tok in enumerate(tokenized_text):
            if tok == mtok: masked_index = i
        indexed_tokens = tokenizer(tokenized_text, add_special_tokens=False, return_tensors="pt").input_ids
        tokens_tensor = indexed_tokens
        yield tokens_tensor, masked_index,tokenized_text


def get_predictions(input_sents,model,tokenizer,k=5,bert=True):
    token_preds = []
    tok_probs = []

    for tokens_tensor, mi, tokensized_text in prep_input(input_sents,tokenizer,bert=bert):
        tokens_tensor = tokens_tensor.to(device)

        with torch.no_grad():
            predictions = model(tokens_tensor).logits

        predictions = torch.permute(predictions, (1, 0, 2))
        predicted_tokens = []
        predicted_token_probs = []
        if bert:
            softpred = torch.softmax(predictions[0,mi],0)
        else:
            softpred = torch.softmax(predictions[0, mi, :],0)

        top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
        top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
        top_tok_preds = tokenizer.convert_ids_to_tokens(top_inds)
        if not bert:
            top_tok_preds = [re.sub('\<\/w\>','',e) for e in top_tok_preds]

        token_preds.append(top_tok_preds)
        tok_probs.append(top_probs)
    return token_preds,tok_probs

def get_probabilities(input_sents,tgtlist,model,tokenizer,bert=True):
    token_probs = []
    for i,(tokens_tensor, mi,_) in enumerate(prep_input(input_sents,tokenizer,bert=bert)):
        tokens_tensor = tokens_tensor.to(device)

        with torch.no_grad():
            predictions = model(tokens_tensor).logits
        predictions = torch.permute(predictions, (1, 0, 2))
        tgt = tgtlist[i]
        if bert:
            softpred = torch.softmax(predictions[0,mi],0)
        else:
            softpred = torch.softmax(predictions[0, mi, :],0)
        try:
            tgt_ind = tokenizer.convert_tokens_to_ids([tgt])[0]
        except:
            this_tgt_prob = np.nan
        else:
            this_tgt_prob = softpred[tgt_ind].item()
        token_probs.append(this_tgt_prob)
    return token_probs


if __name__ == "__main__":
    raise NotImplementedError
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--modeldir", default=None, type=str, required=True)
    args = parser.parse_args()
    get_predictions_roberta(args.input_file)
