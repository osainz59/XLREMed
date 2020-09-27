import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import product
import argparse
import gzip


def get_text(root):
    sentences = []
    sent, sent_id = "", 1
    offset_diff = 0
    for token in root.findall('./text/wf'):
        if int(token.get('sent')) > sent_id:
            sentences.append(sent.strip())
            sent_id += 1
            sent = ""
            offset_diff = max(0, int(token.get('offset')))
        token.set('offset', str(int(token.get('offset')) - offset_diff))
        if int(token.get('offset')) > len(sent):
            sent += " "
        sent += token.text
    sentences.append(sent.strip())
    return sentences

def get_references(root):
    references = set(token.get('reference') 
                    for token in root.findall('.//externalRef[@resource="UMLS-2010AB!"]'))

    return references

def get_span_offset(root, span):
    """ Returns (sentence_id, start_offset, end_offset)
    """
    offsets = []
    for word in span.findall('target'):
        wid = word.get('id')
        wf = root.find(f'.//wf[@wid="{wid}"]')
        sent_id = int(wf.get('sent'))
        offsets.append( (int(wf.get('offset')), int(wf.get('offset')) + int(wf.get('lenght'))) ) # lenght wtf

    return sent_id - 1, offsets[0][0], offsets[-1][-1]

def get_terms_by_reference(root, references):
    entities = defaultdict(list)
    for reference in references:
        for token in root.findall(f'.//externalRef[@reference="{reference}"]/../../..'):
            entities[reference].append( (token.get('lemma'), ) + get_span_offset(root, token.find('span')))
    return entities

def get_crosslingual_instances(en, es, paired_references, en_text, es_text):
    es_terms = get_terms_by_reference(es, paired_references)
    en_terms = get_terms_by_reference(en, paired_references)

    instances = {
        'en': {},
        'es': {}
    }
    for t1, t2 in product(paired_references, paired_references):
        if t1 == t2:
            continue
        # Spanish pairs
        es_pairs = set()
        for s1, s2 in product(es_terms[t1], es_terms[t2]):
            # Check for term pairs that are in the same sentence
            if s1[1] in [0, 1] or s2[1] in [0, 1] or s1[1] != s2[1]:
                continue
            # Check for term pairs that doesn't overlap in the sentence
            if (s1[-2] >= s2[-2] and s1[-2] <= s2[-1]) or \
                (s2[-2] >= s1[-2] and s2[-2] <= s1[-1]) or \
                (s1[-1] >= s2[-2] and s1[-1] <= s2[-1]) or \
                (s2[-1] >= s1[-2] and s2[-1] <= s1[-1]):
                continue
            es_pairs.add((s1, s2))
        if len(es_pairs):
            instances['es'][(t1, t2)] = es_pairs

        # English pairs
        en_pairs = set()
        for s1, s2 in product(en_terms[t1], en_terms[t2]):
            # Check for term pairs that are in the same sentence
            if s1[1] in [0, 1] or s2[1] in [0, 1] or s1[1] != s2[1]:
                continue
            # Check for term pairs that doesn't overlap in the sentence
            if (s1[-2] >= s2[-2] and s1[-2] <= s2[-1]) or \
                (s2[-2] >= s1[-2] and s2[-2] <= s1[-1]) or \
                (s1[-1] >= s2[-2] and s1[-1] <= s2[-1]) or \
                (s2[-1] >= s1[-2] and s2[-1] <= s1[-1]):
                continue
            en_pairs.add((s1, s2))
        if len(en_pairs):
            instances['en'][(t1, t2)] = en_pairs
    
    positive_instance_keys = set(instances['en'].keys()) & set(instances['es'].keys())

    positive_instances = []
    for key in positive_instance_keys:
        en_inst = instances['en'][key]
        es_inst = instances['es'][key]
 
        for en_, es_ in product(en_inst, es_inst):
            # If the subj comes first
            en_sentence = en_text[en_[0][1]]
            if en_[0][-2] < en_[1][-2]:
                en_sentence = en_sentence[:en_[0][-2]] + '[E1S]' + en_sentence[en_[0][-2]:en_[0][-1]] + '[E1E]' + \
                              en_sentence[en_[0][-1]:en_[1][-2]] + '[E2S]' + en_sentence[en_[1][-2]:en_[1][-1]] + '[E2E]' + \
                              en_sentence[en_[1][-1]:]
            else:
                en_sentence = en_sentence[:en_[1][-2]] + '[E2S]' + en_sentence[en_[1][-2]:en_[1][-1]] + '[E2E]' + \
                              en_sentence[en_[1][-1]:en_[0][-2]] + '[E1S]' + en_sentence[en_[0][-2]:en_[0][-1]] + '[E1E]' + \
                              en_sentence[en_[0][-1]:]
            
            es_sentence = es_text[es_[0][1]]
            if en_[0][-2] < en_[1][-2]:
                es_sentence = es_sentence[:es_[0][-2]] + '[E1S]' + es_sentence[es_[0][-2]:es_[0][-1]] + '[E1E]' + \
                              es_sentence[es_[0][-1]:es_[1][-2]] + '[E2S]' + es_sentence[es_[1][-2]:es_[1][-1]] + '[E2E]' + \
                              es_sentence[es_[1][-1]:]
            else:
                es_sentence = es_sentence[:es_[1][-2]] + '[E2S]' + es_sentence[es_[1][-2]:es_[1][-1]] + '[E2E]' + \
                              es_sentence[es_[1][-1]:es_[0][-2]] + '[E1S]' + es_sentence[es_[0][-2]:es_[0][-1]] + '[E1E]' + \
                              es_sentence[es_[0][-1]:]

            # Instance = (subj_concept, obj_concept, subj_en_lemma, obj_en_lemma, subj_es_lemma, obj_es_lemma, en_sentence, es_sentence, class)
            positive_instances.append( key + key + (en_[0][0], en_[1][0], es_[0][0], es_[1][0], en_sentence, es_sentence, 1) )

    # Remove duplicated
    positive_instances_nodup = []
    sents = set()
    for instance in positive_instances:
        en_sent, es_sent = instance[-3:-1]
        if (en_sent, es_sent) in sents:
            continue
        sents.add((en_sent, es_sent))
        positive_instances_nodup.append(instance)

    #print(positive_instance_keys)
    en_neg_inst = set(instances['en'].keys())# - positive_instance_keys
    es_neg_inst = set(instances['es'].keys())# - positive_instance_keys

    negative_instances_keys = []
    for en_pair, es_pair in product( en_neg_inst, es_neg_inst ):
        if any(t in es_pair for t in en_pair) and not all(t in es_pair for t in en_pair):
            negative_instances_keys.append( (en_pair, es_pair) )

    negative_instances = []
    for en_key, es_key in negative_instances_keys:
        en_inst = instances['en'][en_key]
        es_inst = instances['es'][es_key]
 
        for en_, es_ in product(en_inst, es_inst):
            #print(en_, es_)
            # If the subj comes first
            en_sentence = en_text[en_[0][1]]
            if en_[0][-2] < en_[1][-2]:
                en_sentence = en_sentence[:en_[0][-2]] + '[E1S]' + en_sentence[en_[0][-2]:en_[0][-1]] + '[E1E]' + \
                              en_sentence[en_[0][-1]:en_[1][-2]] + '[E2S]' + en_sentence[en_[1][-2]:en_[1][-1]] + '[E2E]' + \
                              en_sentence[en_[1][-1]:]
            else:
                en_sentence = en_sentence[:en_[1][-2]] + '[E2S]' + en_sentence[en_[1][-2]:en_[1][-1]] + '[E2E]' + \
                              en_sentence[en_[1][-1]:en_[0][-2]] + '[E1S]' + en_sentence[en_[0][-2]:en_[0][-1]] + '[E1E]' + \
                              en_sentence[en_[0][-1]:]
            
            es_sentence = es_text[es_[0][1]]
            if en_[0][-2] < en_[1][-2]:
                es_sentence = es_sentence[:es_[0][-2]] + '[E1S]' + es_sentence[es_[0][-2]:es_[0][-1]] + '[E1E]' + \
                              es_sentence[es_[0][-1]:es_[1][-2]] + '[E2S]' + es_sentence[es_[1][-2]:es_[1][-1]] + '[E2E]' + \
                              es_sentence[es_[1][-1]:]
            else:
                es_sentence = es_sentence[:es_[1][-2]] + '[E2S]' + es_sentence[es_[1][-2]:es_[1][-1]] + '[E2E]' + \
                              es_sentence[es_[1][-1]:es_[0][-2]] + '[E1S]' + es_sentence[es_[0][-2]:es_[0][-1]] + '[E1E]' + \
                              es_sentence[es_[0][-1]:]

            negative_instances.append( en_key + es_key + (en_[0][0], en_[1][0], es_[0][0], es_[1][0], en_sentence, es_sentence, 0) )

    # Remove duplicated
    negative_instances_nodup = []
    sents = set()
    for instance in negative_instances:
        en_sent, es_sent = instance[-3:-1]
        if (en_sent, es_sent) in sents:
            continue
        sents.add((en_sent, es_sent))
        negative_instances_nodup.append(instance)


    return positive_instances_nodup + negative_instances_nodup
        

def write_instances(out_file, instances):
    for instance in instances:
        inst_str = '\t'.join(str(elem) for elem in instance) + '\n'
        out_file.write(inst_str.encode('utf-8'))


def main(folder_path, output_file):
    en_path = os.path.join(folder_path, 'en')
    es_path = os.path.join(folder_path, 'es')

    with gzip.open(output_file, 'wb') as out_f:
        # Write header
        out_f.write( ('\t'.join(["en_subj_concept", "en_obj_concept", "es_subj_concept", \
            "es_obj_concept", "subj_en_lemma", "obj_en_lemma", "subj_es_lemma", \
            "obj_es_lemma", "en_sentence", "es_sentence", "class"]) + '\n').encode('utf-8') )

        errors = []
        for file_path in os.listdir(en_path):
            print(file_path)
            try:
                en_file = os.path.join(en_path, file_path)
                es_file = os.path.join(es_path, file_path.replace('en', 'es'))

                # Parse both XML files
                en = ET.parse(en_file)
                es = ET.parse(es_file)

                en_text = get_text(en)
                es_text = get_text(es)

                en_references = get_references(en)
                es_references = get_references(es)

                paired_references = en_references & es_references

                instances = get_crosslingual_instances(en, es, paired_references, en_text, es_text)

                write_instances(out_f, instances)
            except Exception as e:
                errors.append((file_path, e))

        print(errors)
        print("Number of errors:", len(errors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTB dataset generator for MedlineAbstracts.')
    parser.add_argument('folder_path', type=str, help='MedlineAbstracts corpus path.')
    parser.add_argument('--output_file', type=str, default='xlmedline_mtb.tab.gz', help='The output file containing the dataset')

    args = parser.parse_args()
    main(**vars(args)) 
