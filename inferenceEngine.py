from typing import List
from collections import deque

class Request: # what we pass into inference engine
    def __init__(self, id, prompt, max_tokens, generated_tokens, completed):
        self.id = id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.generated_tokens = generated_tokens
        self.completed = completed # by default
    

class InferenceEngine:
    def __init__(self, model):
        self.queue = deque()
        self.results = []
        self.model = model # Model class (Grok 4)

    def enque(self, request: Request) -> None:
        assert isinstance(request, Request)
        self.queue.append(request)
       
    
    def forward_pass(self, prompts: List[str], num_tokens) -> List[List[str]]: 
        """
        Takes in list of prompts, generates forward pass of num_tokens for each
        Returns List[String]
        """

        generated = self.model.generate(prompts, num_tokens) # model takes in prompts and num_tokens -> (parrallelize by B and T)

        return generated

    def processRequests(self, batch_size: int):
        """
        Process requests at batch_size till we have no more requests
        """
        while self.queue: # while there are requests to process
            batch = []
            while self.queue and len(batch) < batch_size:
                batch.append(self.q.popleft())
            
            if not batch:
                return self.results
            prompts = [b.prompt for b in batch]
            generate = self.forward_pass(prompts, 1) # generate 1 token at a time

            for idx, req in enumerate(batch): 
                needed = req.max_tokens - len(req.generated_tokens)
                req.generated_tokens.extend(generate[idx][:needed]) 

                if len(req.generated_tokens) == req.max_tokens:
                    req.completed = True
                    self.results.append((req.id, req.generated_tokens))
                else:
                    self.enque(req)

        return self.results



    




