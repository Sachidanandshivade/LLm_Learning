import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ============================================
# Training Data — Q&A format
# ============================================
text = """
Q: What is your name? A: My name is Sachidanand Shivade.
Q: Who are you? A: I am Sachidanand Shivade, a Full Stack Developer.
Q: What is your role? A: I am a Full Stack Developer.
Q: What are your skills? A: My skills are Java, Spring Boot, React, MySQL, MongoDB, JavaScript, Python, Docker, JWT, REST APIs, Hibernate, Spring Security, TailwindCSS.
Q: What technologies do you know? A: I know Java, Spring Boot, React, MySQL, MongoDB, JavaScript, Python, Docker, JWT, REST APIs, Hibernate, Spring Security, TailwindCSS.
Q: What projects have you built? A: I built a Car Pooling Platform, Portfolio, Contact Manager, and Adaptive Learning Platform.
Q: Tell me about your Car Pooling Platform. A: I built a full stack carpooling platform with JWT authentication, role based access control, 10 plus RESTful APIs, Spring Boot, JPA, React, MySQL.
Q: Tell me about your Portfolio. A: I built a responsive portfolio on Vercel using React, TailwindCSS, with animations and resume download feature.
Q: Tell me about your Contact Manager. A: I built a Contact Manager SPA using React, React Router, TailwindCSS with add, search, delete features deployed on Vercel.
Q: Tell me about your Adaptive Learning Platform. A: I built an Adaptive Learning System with AI tutoring using Cerebras LLaMA 3.1, microservices backend with Node.js, Docker, MongoDB, React frontend.
Q: What is your education? A: I am pursuing Bachelor of Technology in CSIT from Reva University 2022 to 2026 with CGPA of 8.6.
Q: Where do you study? A: I study at Reva University, pursuing Bachelor of Technology in CSIT with CGPA of 8.6.
Q: Do you have internship experience? A: Yes, I am working as Software Development Intern at Kodnest Technologies, Bengaluru since March 2026.
Q: Where do you work? A: I work as Software Development Intern at Kodnest Technologies, Bengaluru since March 2026.
Q: What did you do at Kodnest? A: At Kodnest I developed proficiency in Java, MySQL, Spring Boot, worked on real world development workflows, debugging, and team collaboration.
Q: What are your certifications? A: I have Advanced Java certification from Udemy and JavaScript ES6 certification from Udemy.
Q: What is your email? A: My email is shivade.sachidanand@gmail.com.
Q: What is your phone number? A: My phone number is 9353947020.
Q: What is your GitHub? A: My GitHub is github.com/Sachidanandshivade.
Q: What is your LinkedIn? A: My LinkedIn is sachidanand-shivade-197a802a0.
Q: What databases do you use? A: I use MySQL and MongoDB.
Q: What frontend technologies do you know? A: I know React, HTML5, CSS3, TailwindCSS, JavaScript.
Q: What backend technologies do you know? A: I know Spring Boot, JPA, REST APIs, Hibernate, Spring Security, JWT Authentication, Node.js.
Q: What are your soft skills? A: My soft skills are Problem Solving, Analytical Thinking, Team Collaboration, Adaptability, Continuous Learning, Time Management.
Q: What languages do you speak? A: I speak English, Hindi, and Kannada.
Q: What is your LeetCode rank? A: My LeetCode rank is 1,529,106.
Q: What is your CGPA? A: My CGPA is 8.6 at Reva University.
Q: Are you a developer? A: Yes, I am a Full Stack Developer skilled in React and Spring Boot.
Q: What is your tech stack? A: My tech stack is Java, Spring Boot, React, MySQL, MongoDB, JavaScript, Docker, JWT, REST APIs.
Q: What is your name? A: My name is Sachidanand Shivade.
Q: Who are you? A: I am Sachidanand Shivade, a Full Stack Developer.
Q: What are your skills? A: My skills are Java, Spring Boot, React, MySQL, MongoDB, JavaScript, Python, Docker.
Q: What projects have you built? A: I built Car Pooling Platform, Portfolio, Contact Manager, Adaptive Learning Platform.
Q: What is your education? A: Bachelor of Technology in CSIT from Reva University 2022 to 2026, CGPA 8.6.
Q: Where do you work? A: Software Development Intern at Kodnest Technologies Bengaluru since March 2026.
"""

# ============================================
# Tokenizer
# ============================================
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [char_to_id[ch] for ch in s]

def decode(ids):
    return "".join([id_to_char[i] for i in ids])

print(f"Text length: {len(text)} characters")
print(f"Vocab size:  {vocab_size}")

data = torch.tensor(encode(text))

# ============================================
# Batch Generator
# ============================================
block_size = 64

def get_batch(data, block_size, batch_size=8):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ============================================
# Model — Scaled Up!
# ============================================
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.embed_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return self.out(torch.matmul(weights, V))


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
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


# Scaled up — bigger model!
embed_dim  = 128
num_blocks = 6
max_seq_len = block_size

model = GPT(vocab_size, embed_dim, num_blocks, max_seq_len)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# ============================================
# Training — more steps!
# ============================================
print(f"\n=== Training ===")
for step in range(10000):
    x, y = get_batch(data, block_size)
    logits = model(x)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if step % 1000 == 0:
    #     print(f"Step {step:5d}: loss = {loss.item():.4f}")

print("\n✅ Training complete!")

# Save the model!
torch.save({
    'model_state': model.state_dict(),
    'char_to_id': char_to_id,
    'id_to_char': id_to_char,
    'vocab_size': vocab_size,
    'embed_dim': embed_dim,
    'num_blocks': num_blocks,
    'block_size': block_size,
}, 'sachi_gpt.pth')
print("Model saved to sachi_gpt.pth!")

# ============================================
# Smart Generation — Q&A style
# ============================================
def generate_answer(model, question, max_new_tokens=100):
    model.eval()
    # Format as Q&A so model knows to answer
    prompt = f"Q: {question} A:"
    token_ids = torch.tensor(encode(prompt)).unsqueeze(0)

    for _ in range(max_new_tokens):
        input_ids = token_ids[:, -block_size:]
        with torch.no_grad():
            logits = model(input_ids)
        last_logits = logits[0, -1, :]
        # Use temperature for better generation
        last_logits = last_logits / 0.8
        probs = F.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_char = id_to_char[next_id]
        # Stop at next Q: (new question started)
        if next_char == 'Q' and len(id_to_char) > 0:
            break
        token_ids = torch.cat([
            token_ids,
            torch.tensor([[next_id]])
        ], dim=1)

    # Extract just the answer part
    full_output = decode(token_ids[0].tolist())
    answer = full_output[len(f"Q: {question} A:"):]
    return answer.strip()

# ============================================
# Test it!
# ============================================
print("\n=== Testing Your GPT ===\n")

questions = [
    "What is your name?",
    "What are your skills?",
    "What projects have you built?",
    "Where do you work?",
    "What is your education?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {generate_answer(model, q)}")
    print()