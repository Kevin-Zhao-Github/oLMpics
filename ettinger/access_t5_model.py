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
import transformers


device = "cuda" if torch.cuda.is_available() else "cpu"
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", '']  # nltk

def load_model(modeldir):
    assert "t5" in modeldir
    tokenizer = AutoTokenizer.from_pretrained(modeldir)
    # Load pre-trained model (weights)
    # model = AutoModelForMaskedLM.from_pretrained(modeldir).to(device)
    model = transformers.T5ForConditionalGeneration.from_pretrained(modeldir).to(device)
    model.eval()
    return model,tokenizer


def prep_input(input_sents, tokenizer,bert=True):
    assert "t5" in tokenizer.name_or_path
    for sent in input_sents:
        masked_index = None
        text = []
        # mtok = '<extra_id_0>'# '[MASK]'
        if not bert:
            sent = re.sub('\[MASK\]','X',sent)
            mtok = 'x</w>'
        #if bert: text.append('[CLS]')
        text += sent.strip().split()
        if text[-1] != '.': text.append('.')
        text.append('</s>')
        #if bert: text.append('[SEP]')
        text = ' '.join(text)
        tokenized_text = tokenizer.tokenize(text)
        # for i,tok in enumerate(tokenized_text):
        #     if tok == mtok: masked_index = i
        masked_index = 1
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        yield tokens_tensor, masked_index,tokenized_text


def get_predictions(input_sents,model,tokenizer,k=5,bert=True):
    assert "t5" in tokenizer.name_or_path
    token_preds = []
    tok_probs = []

    for tokens_tensor, mi, tokensized_text in prep_input(input_sents,tokenizer,bert=bert):
        tokens_tensor = tokens_tensor.to(device)

        print(tokenizer.decode(tokens_tensor[0]))

        with torch.no_grad():
            decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            predictions = model(tokens_tensor, decoder_input_ids=decoder_ids).logits

        predicted_tokens = []
        predicted_token_probs = []
        softpred = torch.softmax(predictions[0, mi], 0)

        # top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
        top_inds = torch.argsort(softpred,descending=True)[:180 + k].cpu().numpy()
        top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
        """
        print("herehere")
        print(tokenizer.decode(tokens_tensor[0]))
        asdf_inds = torch.argsort(softpred,descending=True)[:15].cpu().numpy()
        for id in asdf_inds:
            print(tokenizer.decode([id]))

        print("----")
        with torch.no_grad():
            print(tokenizer.decode(tokens_tensor[0]))
            outputs = model.generate(tokens_tensor, num_beams=50, num_return_sequences=20)
            for asd in range(20):
                print(tokenizer.decode(outputs[asd])) #, skip_special_tokens=True))

        assert False
        """
        # top_tok_preds = tokenizer.decode(top_inds) # tokenizer.convert_ids_to_tokens(top_inds)
        # # if not bert:
        # # top_tok_preds = [re.sub('\<\/w\>','',e) for e in top_tok_preds]
        # # print(top_tok_preds)
        top_tok_preds = []
        i = 0
        while len(top_tok_preds) < k:
            if tokenizer.decode(top_inds[i]).strip() not in stop_words:
                top_tok_preds.append(tokenizer.decode(top_inds[i]).strip())
            i += 1
        token_preds.append(top_tok_preds)
        tok_probs.append(top_probs)

    return token_preds,tok_probs

def get_probabilities(input_sents,tgtlist,model,tokenizer,bert=True):
    token_probs = []
    for i,(tokens_tensor, mi,_) in enumerate(prep_input(input_sents,tokenizer,bert=bert)):
        tokens_tensor = tokens_tensor.to(device)

        with torch.no_grad():
            decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            predictions = model(tokens_tensor, decoder_input_ids=decoder_ids).logits

        tgt = tgtlist[i]
        softpred = torch.softmax(predictions[0,mi],0)
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
