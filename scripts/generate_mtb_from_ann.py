import sys
import locale
from itertools import combinations
from random import shuffle
import string

class BRATReader:

    def __init__(self, ann_file, txt_file, encoding=None, length_threshold=5):

        self.ann_file = ann_file
        self.txt_file = txt_file
        self.encoding = encoding if encoding else locale.getpreferredencoding()
        self.order = [0, 1]
        self.length_threshold = length_threshold

    def _handle_entity(self, line, prev_length):
        line = line.decode(self.encoding).strip().split('\t')
        args = line[1:]

        text = args[-1].lower()
        label = args[0].split()[0]

        positions = " ".join(args[0].split()[1:])
        positions = [list(map(int, word.strip().split())) for word in positions.split(';')]
        for i in range(len(positions)):
            positions[i][0] -= prev_length
            positions[i][1] -= prev_length
        output = {
            'positions': positions,
            'text': text,
            'prev_length': prev_length
        }
        #print(output, prev_length)
        return output

    def _yield_entities(self, entities, line):
        for e1, e2 in combinations(entities, 2):
            if e1['text'] == e2['text']:
                continue

            shuffle(self.order)
            subj, obj = (e1, e2) if all(x==y for x, y in zip(self.order, [0, 1])) else (e2, e1)
            subj_start, subj_end = subj['positions'][0][0], subj['positions'][-1][-1]
            obj_start, obj_end = obj['positions'][0][0], obj['positions'][-1][-1]

            if subj_start < obj_start:
                text = line[:subj_start] + '[E1S]' + line[subj_start:subj_end] + '[E1E]' + line[subj_end:obj_start] + '[E2S]' + line[obj_start:obj_end] + '[E2E]' + line[obj_end:]
            else:
                text = line[:obj_start] + '[E2S]' + line[obj_start:obj_end] + '[E2E]' + line[obj_end:subj_start] + '[E1S]' + line[subj_start:subj_end] + '[E1E]' + line[subj_end:]

            #yield subj['text'], obj['text'], text 
            yield subj['text'], obj['text'], text 

    def _filter_entity(self, entity):
        text = entity['text']
        if len(text) <= self.length_threshold:
            return True
        if any(c in text for c in string.digits):
            return True
        
        return False

    def __call__(self):
        prev_length = length = 0
        entities = []
        with open(self.ann_file, 'rb') as ann_f, open(self.txt_file, 'rb') as txt_f:
            text = next(txt_f).decode(locale.getpreferredencoding()).rstrip()
            length += len(text) + 1
            for i, line in enumerate(ann_f):
                entity = self._handle_entity(line, prev_length)

                if entity['positions'][0][0] + prev_length >= length:
                    for e1, e2, instance in self._yield_entities(entities, text):
                        print(f"{len(text)}\t{e1}\t{e2}\t{instance}")

                    while entity['positions'][0][0] + prev_length >= length:
                        entities = []
                        text = next(txt_f).decode(locale.getpreferredencoding()).rstrip()
                        prev_length = length
                        length += len(text) + 1
                        entity = self._handle_entity(line, prev_length)
                
                if self._filter_entity(entity):
                    continue
                entities.append(entity)


                if i > 50:
                    break


if __name__ == "__main__":
    reader = BRATReader(sys.argv[1], sys.argv[2])
    reader()
