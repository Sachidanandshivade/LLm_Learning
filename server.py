from fastapi import FastAPI # Python framework for building web Api's
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # python library that validated the data
import torch
import torch.nn as nn
import torch.nn.functional as F

app = FastAPI()

# Allow requests from your React portfolio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allow requests from ANY website
    allow_methods=["*"],# allow GET, POST, etc
    allow_headers=["*"],  # allow any headers

)

# ============================================
# Load your trained GPT
# ============================================
class SelfAttention(nn.Module): # gets all pytorch features (kind of inheritence)

    # Every word asks every other word
    # "how relevant are you to me?" and
    # collects information weighted by relevance.

    def __init__(self, embed_dim): # embed_div is size og each token vector
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)


    #forward() defines what happens
    # when data passes through your layer.


    #forward  → data flows IN  → predictions come OUT
    #backward → gradients flow back → weights get updated
    def forward(self, x):
        seq_len = x.shape[1]
        Q = self.W_q(x)  #what i am searching for
        K = self.W_k(x) #what do i contain
        V = self.W_v(x) #what do i share
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.embed_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return self.out(torch.matmul(weights, V))

# SelfAttention:
# "each token looks at OTHER tokens"
# → communication between tokens
#
# FeedForward:
# "each token thinks about what it learned"
# → computation within each token

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), #Gaussian Error Linear Unit
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1) #It prevents overfitting — when model memorizes training data instead of learning patterns.
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_blocks, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.shape[1]
        tok = self.token_emb(token_ids)
        positions = torch.arange(seq_len)
        pos = self.pos_emb(positions)
        x = tok + pos
        x = self.blocks(x)
        x = self.norm(x)
        return self.lm_head(x)


# Load saved model
checkpoint = torch.load('sachi_gpt.pth', map_location='cpu')
char_to_id = checkpoint['char_to_id']
id_to_char = checkpoint['id_to_char']
vocab_size  = checkpoint['vocab_size']
embed_dim   = checkpoint['embed_dim']
num_blocks  = checkpoint['num_blocks']
block_size  = checkpoint['block_size']

model = GPT(vocab_size, embed_dim, num_blocks, block_size)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print("✅ Model loaded successfully!")

# ============================================
# Helper functions
# ============================================
def encode(s):
    return [char_to_id.get(ch, 0) for ch in s]

def decode(ids):
    return "".join([id_to_char.get(i, '') for i in ids])

def generate_answer(question, max_new_tokens=150):
    prompt = f"Q: {question} A:"
    token_ids = torch.tensor(encode(prompt)).unsqueeze(0)

    for _ in range(max_new_tokens):
        input_ids = token_ids[:, -block_size:]
        with torch.no_grad():
            logits = model(input_ids)
        last_logits = logits[0, -1, :] / 0.8
        probs = F.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_char = id_to_char.get(next_id, '')
        if next_char == 'Q':
            break
        token_ids = torch.cat([
            token_ids,
            torch.tensor([[next_id]])
        ], dim=1)

    full_output = decode(token_ids[0].tolist())
    answer = full_output[len(f"Q: {question} A:"):]
    return answer.strip()

# ============================================
# API Endpoint
# ============================================
class Question(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "Sachi GPT is running!"}

@app.post("/chat")
def chat(q: Question):
    answer = generate_answer(q.message)
    return {"answer": answer}