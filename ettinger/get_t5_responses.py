import argparse
import copy
from io import open
import random
import os
import re
import scipy
import scipy.stats
import numpy as np

import access_t5_model as tp


def get_model_responses(inputlist, tgtlist, modeliname, model, tokenizer, k=5, bert=True):
    top_preds, top_probs = tp.get_predictions(inputlist, model, tokenizer, k=k, bert=bert)
    tgt_probs = tp.get_probabilities(inputlist, tgtlist, model, tokenizer, bert=bert)

    return top_preds, top_probs, tgt_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", default=None, type=str)
    parser.add_argument("model_name_or_path", default=None, type=str)
    args = parser.parse_args()

    testlist = ['cprag','role', 'negsimp','negnat']

    print('LOADING MODELS')
    model, tokenizer = tp.load_model(args.model_name_or_path)

    k = 5

    for testname in testlist:
        inputlist = []
        tgtlist = []
        print("1 check")
        with open(os.path.join(args.data_dir,testname+'-contextlist')) as cont:
            for line in cont:
                assert line.strip()[-7:] == " [MASK]"
                t5_mask_line = line.strip()[:-7] + " <extra_id_0>"
                inputlist.append(t5_mask_line)

        with open(os.path.join(args.data_dir,testname+'-targetlist')) as comp:
            for line in comp:
                tgtlist.append(line.strip())

        print("2 check")
        top_preds, top_probs, tgt_probs = get_model_responses(inputlist, tgtlist, args.model_name_or_path, model, tokenizer, k=k)
        with open(args.data_dir + '/modelpreds-%s-%s'%(testname, args.model_name_or_path), 'w') as pred_out:
            for i,preds in enumerate(top_preds):
                # words = preds.split(' ')
                # words = list(filter(None, words))
                # pred_out.write(' '.join(words))
                pred_out.write(' '.join(preds))
                pred_out.write('\n')

        with open(args.data_dir + '/modeltgtprobs-%s-%s'%(testname, args.model_name_or_path), 'w') as prob_out:
            for i,prob in enumerate(tgt_probs):
                prob_out.write(str(prob))
                prob_out.write('\n')
