#!/usr/bin/env python3
"""
ðŸš€ PRÃ‰-ENTRAÃŽNEMENT HessGPT - GPT-2 TOKENIZER
âœ… 20M tokens Ã— 3 Ã©poques
âœ… GPT-2 tokenizer (50k vocab, stable)
âœ… Format simple pour SFT
âœ… GARANTIE de fonctionner !
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
import time
import math
from datasets import load_dataset
from tqdm import tqdm

sys.path.append('./Core/Model')
sys.path.append('./Core/Training')

from HessGpt import HessGPT
from transformers import GPT2Tokenizer

print("="*60)
print("ðŸš€ PRÃ‰-ENTRAÃŽNEMENT HessGPT - GPT-2 TOKENIZER")
print("="*60)

# ============================================
# GPU
# ============================================
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"\nâœ… GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print(f"âœ… PyTorch: {torch.__version__}")
else:
    device = 'cpu'
    print("âŒ GPU non disponible!")
    sys.exit(1)

# ============================================
# CONFIG (100M PARAMS)
# ============================================
CONFIG = {
    # Architecture - 100M params
    'vocab_size': 50257,  # GPT-2 vocab
    'embed_dim': 768,     # Réduit (512 → 768)
    'num_heads': 12,      # Standard
    'num_layers': 12,      # Réduit (12 → 8)
    'max_seq_len': 1024,
    'dropout': 0.1,
    
    # Training - Optimisé pour 100M
    'batch_size': 32,              
    'num_epochs': 5,
    'learning_rate': 2e-4,
    'warmup_steps': 500,
    'gradient_accumulation': 4,    # Effective batch = 48
    'max_grad_norm': 1.0,
    
    # Data
    'max_tokens': 500_000_000,
    'min_text_length': 100,
    
    # Scheduler
    'scheduler_type': 'cosine',
    'min_lr_ratio': 0.1,
    
    # Compilation
    'use_compile': True,
    'compile_mode': 'default',
    'disable_cudagraphs': True,
}

print(f"\nâš™ï¸  Configuration:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

print(f"\nðŸ’¡ TOKENIZER GPT-2:")
print(f"  â†’ Vocab: 50k tokens (stable)")
print(f"  â†’ Pas de tokens spÃ©ciaux complexes")
print(f"  â†’ Format simple et clair")
print(f"  â†’ GARANTIE de fonctionner !")

# ============================================
# TOKENIZER GPT-2
# ============================================
print("\nðŸ”¤ Chargement GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

CONFIG['vocab_size'] = len(tokenizer)
print(f"âœ“ GPT-2 Tokenizer: {len(tokenizer)} tokens")
print(f"âœ“ EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"âœ“ PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# ============================================
# TOKENIZATION SIMPLE
# ============================================
def tokenize_texts(texts, tokenizer):
    """Tokenize simplement sans format spÃ©cial"""
    tokens = []
    for text in texts:
        if len(text) > 10:
            # Tokenize normalement
            encoded = tokenizer.encode(text[:5000], add_special_tokens=False)
            tokens.extend(encoded)
            # Ajouter EOS
            tokens.append(tokenizer.eos_token_id)
    return tokens

# ============================================
# DATASETS
# ============================================
print("\nðŸ“¥ Chargement datasets...")
os.makedirs("data", exist_ok=True)

cache_file = "data/english_500M_gpt2_cache.pt"

if os.path.exists(cache_file):
    print(f"âœ“ Cache: {cache_file}")
    cached = torch.load(cache_file)
    all_tokens = cached['tokens']
    print(f"âœ“ {len(all_tokens):,} tokens ({len(all_tokens)/1e6:.1f}M)")
else:
    all_tokens = []
    
    print(f"\nðŸŒ TÃ©lÃ©chargement FineWeb ({CONFIG['max_tokens']/1e6:.0f}M tokens)...")
    print("â³ 2-3 minutes...")
    
    try:
        fineweb = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        
        batch_texts = []
        doc_count = 0
        estimated_docs = int(CONFIG['max_tokens'] / 500)
        
        pbar = tqdm(fineweb, total=estimated_docs, desc="FineWeb")
        
        for item in pbar:
            if len(all_tokens) >= CONFIG['max_tokens']:
                break
            
            text = item.get('text', '')
            if len(text) > CONFIG['min_text_length']:
                batch_texts.append(text)
                doc_count += 1
                
                if len(batch_texts) >= 500:
                    tokens = tokenize_texts(batch_texts, tokenizer)
                    all_tokens.extend(tokens)
                    batch_texts = []
                    
                    pbar.set_postfix({
                        'docs': f'{doc_count:,}',
                        'tokens': f'{len(all_tokens)/1e6:.1f}M',
                        'progress': f'{len(all_tokens)/CONFIG["max_tokens"]*100:.1f}%'
                    })
        
        if batch_texts:
            tokens = tokenize_texts(batch_texts, tokenizer)
            all_tokens.extend(tokens)
        
        print(f"\nâœ“ Documents: {doc_count:,}")
        
    except Exception as e:
        print(f"âŒ Erreur: {e}")
        sys.exit(1)
    
    if len(all_tokens) > CONFIG['max_tokens']:
        all_tokens = all_tokens[:CONFIG['max_tokens']]
    
    print(f"\nâœ“ Total: {len(all_tokens):,} tokens ({len(all_tokens)/1e6:.1f}M)")
    
    print(f"ðŸ’¾ Sauvegarde cache...")
    torch.save({'tokens': all_tokens, 'config': CONFIG}, cache_file)
    print(f"âœ“ Cache: {cache_file}")

# ============================================
# VÃ‰RIFICATION
# ============================================
print(f"\nðŸ§ª VÃ©rification tokenization:")
sample_text = "Hello, how are you?"
sample_tokens = tokenizer.encode(sample_text)
decoded = tokenizer.decode(sample_tokens)
print(f"  Texte: {sample_text}")
print(f"  Tokens: {sample_tokens}")
print(f"  DÃ©codÃ©: {decoded}")
print(f"âœ… Tokenizer fonctionne correctement!")

# ============================================
# DATASET PYTORCH
# ============================================
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.num_samples = len(tokens) // (seq_len + 1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        
        if end > len(self.tokens):
            chunk = torch.cat([
                self.tokens[start:],
                torch.zeros(end - len(self.tokens), dtype=torch.long)
            ])
        else:
            chunk = self.tokens[start:end]
        
        return chunk[:-1], chunk[1:]

print("\nðŸ“¦ CrÃ©ation dataset...")
train_dataset = TokenDataset(all_tokens, CONFIG['max_seq_len'])
print(f"  â†’ {len(train_dataset):,} sÃ©quences")

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)

num_batches = len(train_loader)
total_steps = (num_batches * CONFIG['num_epochs']) // CONFIG['gradient_accumulation']

print(f"âœ“ Batches/Ã©poque: {num_batches:,}")
print(f"âœ“ Total steps: {total_steps:,}")
print(f"âœ“ Effective batch: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")

# ============================================
# MODÃˆLE
# ============================================
print("\nðŸ¤– CrÃ©ation modÃ¨le...")
model = HessGPT(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'],
    dropout=CONFIG['dropout']
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"âœ“ ParamÃ¨tres: {num_params/1e6:.1f}M")

# ============================================
# COMPILATION
# ============================================
if CONFIG['use_compile']:
    print(f"\nâš¡ Compilation...")
    
    if CONFIG.get('disable_cudagraphs', True):
        torch._inductor.config.triton.cudagraphs = False
    
    try:
        model = torch.compile(
            model,
            mode=CONFIG['compile_mode'],
            fullgraph=False,
            dynamic=False
        )
        print("âœ“ CompilÃ©!")
    except Exception as e:
        print(f"âš ï¸  Erreur: {e}")
        CONFIG['use_compile'] = False

torch.cuda.empty_cache()

# ============================================
# OPTIMIZER & SCHEDULER
# ============================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True
)

def lr_lambda(step):
    if step < CONFIG['warmup_steps']:
        return step / CONFIG['warmup_steps']
    
    progress = (step - CONFIG['warmup_steps']) / (total_steps - CONFIG['warmup_steps'])
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    min_lr = CONFIG['min_lr_ratio']
    return min_lr + (1.0 - min_lr) * cosine_decay

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("\nâœ“ Optimizer: AdamW")
print(f"âœ“ Scheduler: Cosine")

# ============================================
# ENTRAÃŽNEMENT
# ============================================
print("\n" + "="*60)
print("ðŸš€ DÃ‰BUT ENTRAÃŽNEMENT - GPT-2 TOKENIZER")
print("="*60)

tokens_per_second = 5000
total_time = (CONFIG['max_tokens'] * CONFIG['num_epochs']) / tokens_per_second / 60
print(f"â±ï¸  Estimation: ~{total_time:.0f} minutes")
print(f"ðŸ“Š {num_batches:,} batches/Ã©poque Ã— {CONFIG['num_epochs']} Ã©poques")
print("="*60 + "\n")

os.makedirs("checkpoints", exist_ok=True)
start_time = time.time()
compile_done = False

model.train()
train_losses = []
scaler = torch.amp.GradScaler('cuda')

for epoch in range(CONFIG['num_epochs']):
    epoch_loss = 0
    valid_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if CONFIG['use_compile'] and not compile_done and batch_idx == 0 and epoch == 0:
            print("\nâ³ Compilation (1-2 min)...")
        
        with torch.amp.autocast('cuda'):
            logits, loss = model(x, targets=y)
            loss = loss / CONFIG['gradient_accumulation']
        
        if CONFIG['use_compile'] and not compile_done and batch_idx == 1 and epoch == 0:
            compile_done = True
            elapsed = time.time() - start_time
            print(f"âœ“ Compilation: {elapsed/60:.1f} min\n")
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        epoch_loss += loss.item() * CONFIG['gradient_accumulation']
        valid_batches += 1
        train_losses.append(loss.item() * CONFIG['gradient_accumulation'])
        
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            batches_done = batch_idx + 1 + (epoch * num_batches)
            total_batches = num_batches * CONFIG['num_epochs']
            speed = batches_done / elapsed
            eta_seconds = (total_batches - batches_done) / speed if speed > 0 else 0
            
            pbar.set_postfix({
                'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'speed': f'{speed:.1f}it/s',
                'eta': f'{eta_seconds/60:.0f}min'
            })
    
    avg_loss = epoch_loss / max(valid_batches, 1)
    print(f"\nâœ“ Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    
    save_model = model._orig_mod if CONFIG['use_compile'] else model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG,
        'train_losses': train_losses,
        'avg_loss': avg_loss,
    }, f'./checkpoints/hessgpt_gpt2_epoch{epoch+1}.pt')

elapsed = time.time() - start_time

# ============================================
# TEST GÃ‰NÃ‰RATION
# ============================================
print("\n" + "="*60)
print("ðŸŽ‰ TEST GÃ‰NÃ‰RATION")
print("="*60)

eval_model = model._orig_mod if CONFIG['use_compile'] else model
eval_model.eval()

test_prompts = [
    "Hello, how are you",
    "The capital of France is",
    "In Python, a function"
]

@torch.no_grad()
def generate_simple(model, input_ids, max_new_tokens=50):
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.amp.autocast('cuda'):
            logits, _ = model(generated)
        
        next_logits = logits[:, -1, :] / 0.8
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        if generated.size(1) >= CONFIG['max_seq_len']:
            break
    
    return generated

for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = generate_simple(eval_model, input_ids)
    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text[len(prompt):]}")
    print("-" * 60)

# ============================================
# STATS FINALES
# ============================================
print("\n" + "="*60)
print("ðŸ“Š STATISTIQUES FINALES")
print("="*60)
print(f"âœ“ Tokens: {len(all_tokens)/1e6:.1f}M Ã— {CONFIG['num_epochs']} = {len(all_tokens)*CONFIG['num_epochs']/1e6:.0f}M effectifs")
print(f"âœ“ Tokenizer: GPT-2 (50k vocab)")
print(f"âœ“ Loss: {train_losses[0]:.4f} â†’ {train_losses[-1]:.4f}")
print(f"âœ“ AmÃ©lioration: {((train_losses[0]-train_losses[-1])/train_losses[0]*100):.1f}%")
print(f"âœ“ Temps: {elapsed/60:.1f} min")
print("="*60)

save_model = model._orig_mod if CONFIG['use_compile'] else model
torch.save({
    'model_state_dict': save_model.state_dict(),
    'config': CONFIG,
    'train_losses': train_losses,
    'final_loss': train_losses[-1],
}, './checkpoints/hessgpt_gpt2_final.pt')

print("\nâœ… PRÃ‰-ENTRAÃŽNEMENT TERMINÃ‰!")
print("ðŸ’¾ ModÃ¨le: ./checkpoints/hessgpt_gpt2_final.pt")
print("\nðŸš€ PROCHAINE Ã‰TAPE: SFT avec format simple")
print("="*60)
