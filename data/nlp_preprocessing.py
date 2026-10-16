import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        max_len_pos_sent = max(positive, key=lambda x: len(x.split()))
        max_len_neg_sent = max(negative, key=lambda x: len(x.split()))

        T = max(len(max_len_pos_sent), len(max_len_neg_sent))
        N = len(positive)

        words = {}
        for a in positive:
            sentence = a.split()
            for word in sentence:
                if word not in words:
                    words[word] = 0.0

        for b in negative:
            sentence = b.split()
            for word in sentence:
                if word not in words:
                    words[word] = 0.0
        
        words = dict(sorted(words.items()))
        i: float = 1
        for word in words:
            words[word] = i
            i += 1

        #print(words)

        result = []

        for a in positive:
            sentence = a.split()
            i: float = 1
            word_vect = []
            for word in sentence:
                word_vect.append(words[word])
                i += 1
                if i == T - 1:
                    word_vect.append(0.0)

            result.append(word_vect)

        for b in negative:
            sentence = b.split()
            i: float = 1
            word_vect = []
            for word in sentence:
                word_vect.append(words[word])
                

            result.append(word_vect)

        tensors = [torch.tensor(row, dtype=torch.float32) for row in result]
        result = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)


        return result

        
        

        

        pass
