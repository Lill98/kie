from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch


class KVEmbedding(object):
    def __init__(self, device = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        self.model.eval()
        self.device = device
        self.device = torch.device(self.device)
        self.model.to(self.device)

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embedding(self, l_transcription):
        batch_dict = self.tokenizer(l_transcription, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings
        # embeddings = abs(F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy().mean(axis=1))
        embeddings = abs(F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy())
        return embeddings
        
if __name__ == "__main__":
    kv_embed = KVEmbedding()
    text = "東京都千代1234"
    print(kv_embed.embedding(text))