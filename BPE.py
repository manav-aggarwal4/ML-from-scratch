from typing import List, Tuple
from collections import defaultdict
import heapq
class Tokenizer:
    def __init__(self):
        self.vocab = set()
        self.merges: List[Tuple[str, str]] = []

    def train(self, corpus: List[str], target_vocab_size: int) -> List[str]:
        """
        corpus: List of string values of initial corpus
        target_vocab_size: integer of size of final vocabulary

        iterate, merging the most frequent pair of element's and adding them to the vocab till we reach target_vocab_size
        """

        def findMerge(corpus: List[List[str]]) -> str:
            """
            Does a single merge, return the combined tokens as a str
            """
            freq = defaultdict(int)

            for word in corpus: # for each word, we must look at the char level as we build from the bottom
                for i in range(len(word) - 1):
                    freq[(word[i], word[i + 1])] += 1
            if not freq:
                return None
            
            curr = -1
            mostFreq = None
            for key, val in freq.items():
                if val > curr:
                    curr = val
                    mostFreq = key

            return mostFreq if mostFreq else None

        for word in corpus:
            for char in word:
                self.vocab.add(char)
        
        while len(self.vocab) < target_vocab_size:
            pair = findMerge(corpus)

            if pair is None:
                return None
            self.merges.append(pair)
            first, second = pair
            merge = first + second
            self.vocab.add(merge)
            newCorpus = []

            for word in corpus:
                i = 0
                newTokens = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                        newTokens.append(merge)
                        i += 2
                    else:
                        newTokens.append(word[i])
                        i += 1
                newCorpus.append(newTokens)
            corpus = newCorpus
        return list(self.vocab)

    def encode(self, text: str) -> List[str]:
        """
        Tokenize new text using learned merging
        """
        lst = list(text)
        rank = {pair: i for i, pair in enumerate(self.merges)}
        heap = []
        for i in range(len(lst) - 1):
            pair = (lst[i], lst[i + 1])

            if pair in rank:
                heap.append((rank[pair], i, pair))
        
        heapq.heapify(heap)

        while heap:
            _, idx, pair = heapq.heappop(heap)

            if idx > len(lst) - 2:
                continue # now out of bounds
                
            if (lst[idx], lst[idx + 1]) != pair:
                continue # stale, has already been changed
            
            lst[idx:idx +2] = [lst[idx] + lst[idx + 1]] # do merge 

            if idx - 1 >= 0:
                pair = (lst[idx - 1], lst[idx])

                if pair in rank:
                    heapq.heappush(heap, (rank[pair], idx - 1, pair))
            if idx + 1 < len(lst):
                pair = (lst[idx], lst[idx + 1])
                if pair in rank:
                    heapq.heappush(heap, (rank[pair], idx + 1, pair))




    def decode(self, tokens: List[str]) -> str:
        """
        text: encoded text
        Returns DECODED
        Concatenate all of tokens
        """
        decoded = []
        for token in tokens:
            decoded.append(token)

        return "".join(decoded)



       
