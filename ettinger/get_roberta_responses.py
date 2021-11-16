import re
import os
import argparse
import random
import copy
import numpy as np
import access_model as tp
import scipy
import scipy.stats
from io import open

def get_model_responses_roberta(inputlist,tgtlist,modeliname,model,tokenizer,k=5,bert=False):
    top_preds,top_probs = tp.get_predictions_roberta(inputlist,model,tokenizer,k=k)
    tgt_probs = tp.get_probabilities_roberta(inputlist,tgtlist,model,tokenizer)
    return top_preds,top_probs,tgt_probs, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir", default=None, type=str)
    #parser.add_argument("--bertbase",default=None, type=str)
    #parser.add_argument("--bertlarge",default=None, type=str)
    #parser.add_argument("--robertabase", default=None, type=str)
    #parser.add_argument("--robertalarge", default=None, type=str)
    #parser.add_argument("--distilbertbase",default=None, type=str)
    parser.add_argument("--albertlarge",default=None, type=str)
    args = parser.parse_args()

    testlist = ['cprag','role', 'negsimp','negnat']

    print('LOADING MODELS')
    #bert_base,tokenizer_base = tp.load_model(args.bertbase)
    #bert_large,tokenizer_large = tp.load_model(args.bertlarge)
    #roberta_base, roberta_tokenizer_base = tp.load_model_roberta(args.robertabase)
    #roberta_large, roberta_tokenizer_large = tp.load_model_roberta(args.robertalarge)
    #distilbert_base,distil_tokenizer_base = tp.load_model_roberta(args.distilbertbase)
    albert_large, albert_tokenizer_large = tp.load_model_roberta(args.albertlarge)

    k = 5

    # models = [('bert-base-uncased',bert_base,tokenizer_base),
    # ('bert-large-uncased',bert_large,tokenizer_large),
    # ('roberta-base',roberta_base,roberta_tokenizer_base)]

    # models = [('roberta-base',roberta_base,roberta_tokenizer_base)]
    # models = [('roberta-large',roberta_large,roberta_tokenizer_large)]
    # models = [('distilbert-base-uncased',distilbert_base,distil_tokenizer_base)]
    models = [('albert-large-v2', albert_large,albert_tokenizer_large)]

    for testname in testlist:
        inputlist = []
        tgtlist = []
        with open(os.path.join(args.inputdir,testname+'-contextlist')) as cont:
            for line in cont: inputlist.append(line.strip())
        with open(os.path.join(args.inputdir,testname+'-targetlist')) as comp:
            for line in comp: tgtlist.append(line.strip())
        for modelname,model,tokenizer in models:
            top_preds,top_probs,tgt_probs ,tokenizer= get_model_responses_roberta(inputlist,tgtlist,modelname,model,tokenizer,k=k)
        
            with open(args.inputdir+'/modelpreds-%s-%s'%(testname,modelname),'w', encoding='utf8') as pred_out:  # add encoding='utf8' for distilbert
                for i,preds in enumerate(top_preds):
                    modified_pred = []
                    for each_pred in preds:
                        print(each_pred)
                        strip_id = None
                        inputs = tokenizer(each_pred, return_tensors="pt")
                        each_pred_ids = inputs['input_ids']
                        each_pred_ids = each_pred_ids.reshape(-1)
                        print(each_pred_ids)
                        for j,id in enumerate(each_pred_ids):
                            if id==649:
                                print("check 649")
                                strip_id = j
                                each_pred = each_pred_ids[strip_id+2:-1]
                                each_pred = tokenizer.convert_ids_to_tokens(each_pred)
                                each_pred = ''.join(each_pred)
                                print(each_pred)
                                break
                        modified_pred.append(each_pred)

                    pred_out.write(' '.join(modified_pred))
                    pred_out.write('\n')

            with open(args.inputdir+'/modeltgtprobs-%s-%s'%(testname,modelname),'w') as prob_out:
                for i,prob in enumerate(tgt_probs):
                    prob_out.write(str(prob))
                    prob_out.write('\n')
