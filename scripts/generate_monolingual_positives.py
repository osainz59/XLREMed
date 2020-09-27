import argparse
import gzip
from tqdm import tqdm
import numpy as np
from random import shuffle

import matplotlib.pyplot as plt

from collections import defaultdict

def get_data_from_instance(inst, lang=None):
    inst = inst.decode('utf-8').strip().split('\t')
    if not lang:
        return inst
    elif lang == 'es':
        return inst[2:4] + inst[6:8] + [inst[9]]
    elif lang == 'en':
        return inst[0:2] + inst[4:6] + [inst[8]]
    else:
        raise ValueError('lang argument should have one of the following values [None, "es", "en"]')

def generate_positive_examples(sentences, min_threshold=2, max_threshold=64):
    length = len(sentences)
    if length >= min_threshold:
        sentences = list(sentences.copy())
        shuffle(sentences)
        for sent1, sent2 in zip(sentences[:max_threshold:2], sentences[1:max_threshold+1:2]):
            yield sent1, sent2

def main(in_path, out_path, lang='es', by='lemma'):

    if by not in ['lemma', 'concept']:
        raise ValueError('by argument should be "lemma" or "concept".')

    if lang not in ['es', 'en']:
        raise ValueError('lang argument should be "es" or "en".')

    ents = defaultdict(set)
    ent_pair = defaultdict(set)
    sentences = defaultdict(set)
    all_sentences = set()
    
    with gzip.open(in_path, 'rb') as in_f, gzip.open(out_path, 'wb') as out_f:
        for i, line in tqdm(enumerate(in_f)):
            if not i:   # Skip the header
                continue
            subj_concept, obj_concept, subj_lemma, obj_lemma, sentence = get_data_from_instance(line, lang)
            if sentence in all_sentences:   # To skip repeated sentences that comes from same lemmas with different concepts
                continue
            all_sentences.add(sentence)
            clean_sentence = sentence.replace('[E1S]', '').replace('[E1E]', '').replace('[E2S]', '').replace('[E2E]', '')
            if by is 'concept':
                ent_pair[(subj_concept, obj_concept)].add((sentence, subj_lemma, obj_lemma))
                ents[subj_concept].add((subj_concept, obj_concept))
                ents[obj_concept].add((subj_concept, obj_concept))
                sentences[clean_sentence].add((subj_concept, obj_concept))
            elif by is 'lemma':
                ent_pair[(subj_lemma, obj_lemma)].add((sentence, subj_concept, obj_concept))
                ents[subj_lemma].add((subj_lemma, obj_lemma))
                ents[obj_lemma].add((subj_lemma, obj_lemma))
                sentences[clean_sentence].add((subj_lemma, obj_lemma))
            else:
                continue

        # Compute some statistics about the corpus
        lengths, seq_length, ent_pair_sentence = [], [], []
        for _, value in ent_pair.items():
            lengths.append(len(value))
        for key, value in sentences.items():
            seq_length.append(len(key.split()))
            ent_pair_sentence.append(len(value))

        lengths, seq_length, ent_pair_sentence = np.array(lengths), np.array(seq_length), np.array(ent_pair_sentence)
        print(f"Appearances of each Entity Pair: Min={lengths.min()} Max={lengths.max()} Mean={lengths.mean():.3f} Std: {lengths.std():.3f}")
        print(f"Sentences length: Min={seq_length.min()} Max={seq_length.max()} Mean={seq_length.mean():.3f} Std: {seq_length.std():.3f}")
        print(f"Entity Pairs per sentence: Min={ent_pair_sentence.min()} Max={ent_pair_sentence.max()} Mean={ent_pair_sentence.mean():.3f} Std: {ent_pair_sentence.std():.3f}")

        # Output file
        out_f.write( ('\t'.join(["en_subj_concept", "en_obj_concept", "es_subj_concept", \
            "es_obj_concept", "subj_en_lemma", "obj_en_lemma", "subj_es_lemma", \
            "obj_es_lemma", "en_sentence", "es_sentence", "class"]) + '\n').encode('utf-8') )
        
        # Generate positive instances
        n_instances = 0
        for key, value in tqdm(ent_pair.items(), total=len(ent_pair)):
            for sent1, sent2 in generate_positive_examples(value):
                if by is 'concept':
                    instance = ('\t'.join([key[0], key[1], key[0], key[1], sent1[1], sent1[2], sent2[1], sent2[2], sent1[0], sent2[0], "1"])) + '\n'
                else:
                    instance = ('\t'.join([sent1[1], sent1[2], sent2[1], sent2[2], key[0], key[1], key[0], key[1], sent1[0], sent2[0], "1"])) + '\n'
                out_f.write(instance.encode('utf-8'))
                n_instances += 1

        # Generate as much negative instances than positives
        neg_pairs = set()
        pbar = tqdm(total=n_instances)

        # Generate strong negatives
        while len(ents) > 0 and n_instances >= 0:
            pivots = list(range(len(ents)))
            shuffle(pivots)
            #print(len(pivots), n_instances)
            pivots_to_delete = []
            for pivot in pivots:
                pivot = list(ents.keys())[pivot]
                pairs = list(ents[pivot])
                #print(len(pairs))
                if len(pairs) < 2:
                    # del ents[pivot]
                    pivots_to_delete.append(pivot)
                    continue
                shuffle(pairs)
                p1, p2 = pairs[:2]
                
                if (p1, p2) in neg_pairs:
                    continue

                # Create the new negative instance
                sent1 = list(ent_pair[p1].copy())
                shuffle(sent1)
                sent1 = sent1[0]

                sent2 = list(ent_pair[p2].copy())
                shuffle(sent2)
                sent2 = sent2[0]

                if by is 'concept':
                    if all(sent1[i+1] == sent2[i+1] for i in range(2)) or \
                       all(sent1[i+1] == sent2[-(i+1)] for i in range(2)):
                        ents[pivot].remove(p1)
                        continue
                else:
                    if all(p1[i] == p2[i] for i in range(2)) or \
                       all(p1[i] == p2[-(i+1)] for i in range(2)):
                        ents[pivot].remove(p1)
                        continue

                ents[pivot].remove(p1)
                ents[pivot].remove(p2)
                neg_pairs.add( (p1, p2) )
                n_instances -= 1

                if by is 'concept':
                    instance = ('\t'.join([
                        p1[0], p1[1], p2[0], p2[1], 
                        sent1[1], sent1[2],
                        sent2[1], sent2[2],
                        sent1[0], sent2[0],
                        "0"
                    ]) + '\n').encode('utf-8')
                else:
                    instance = ('\t'.join([
                        sent1[1], sent1[2],
                        sent2[1], sent2[2],
                        p1[0], p1[1], p2[0], p2[1], 
                        sent1[0], sent2[0],
                        "0"
                    ]) + '\n').encode('utf-8')
                out_f.write(instance)
                pbar.update(1)

                if n_instances <= 0:
                    break

            for pivot in pivots_to_delete:
                del ents[pivot]
        
        # Generate weak negative examples
        while len(ent_pair) >= 2 and n_instances >= 0:
            ent_pair_keys = list(ent_pair.keys()).copy()
            shuffle(ent_pair_keys)

            keys_to_delete = []
            for p1, p2 in zip(ent_pair.keys(), ent_pair_keys):
                if p1 == p2 or (p1, p2) in neg_pairs:
                    continue

                sent1 = list(ent_pair[p1].copy())
                shuffle(sent1)
                sent1 = sent1[0]

                sent2 = list(ent_pair[p2].copy())
                shuffle(sent2)
                sent2 = sent2[0]

                if by is 'concept':
                    if all(sent1[i+1] == sent2[i+1] for i in range(2)) or \
                       all(sent1[i+1] == sent2[-(i+1)] for i in range(2)):
                        continue
                else:
                    if all(p1[i] == p2[i] for i in range(2)) or \
                       all(p1[i] == p2[-(i+1)] for i in range(2)):
                        continue

                neg_pairs.add( (p1, p2) )
                neg_pairs.add( (p2, p1) )
                n_instances -= 1

                if by is 'concept':
                    instance = ('\t'.join([
                        p1[0], p1[1], p2[0], p2[1], 
                        sent1[1], sent1[2],
                        sent2[1], sent2[2],
                        sent1[0], sent2[0],
                        "0"
                    ]) + '\n').encode('utf-8')
                else:
                    instance = ('\t'.join([
                        sent1[1], sent1[2],
                        sent2[1], sent2[2],
                        p1[0], p1[1], p2[0], p2[1], 
                        sent1[0], sent2[0],
                        "0"
                    ]) + '\n').encode('utf-8')
                out_f.write(instance)
                pbar.update(1)

        pbar.close()




if __name__ == "__main__":
    main('data/medline/xlmedline_mtb.tab.gz', 'data/medline/esmedline_mtb.concept.tab.gz', lang='es', by='concept')
    main('data/medline/xlmedline_mtb.tab.gz', 'data/medline/enmedline_mtb.concept.tab.gz', lang='en', by='concept')
    main('data/medline/xlmedline_mtb.tab.gz', 'data/medline/esmedline_mtb.lemma.tab.gz', lang='es', by='lemma')
    main('data/medline/xlmedline_mtb.tab.gz', 'data/medline/enmedline_mtb.lemma.tab.gz', lang='en', by='lemma')
