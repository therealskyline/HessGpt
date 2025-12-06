"""
🚀 SFT HessGPT SIMPLIFIÉ - QUALITÉ > QUANTITÉ
✅ Alpaca + Stanford + Dolly + WizardLM + Synthétique 10K
✅ Pas de OpenAssistant (trop conversationnel)
✅ Pas de math/code (focus général)
✅ 6 epochs optimisés
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys, os, time, math, random, json
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset

sys.path.append('./Core/Model')
from HessGpt import HessGPT

print("="*70)
print("🚀 SFT HessGPT SIMPLIFIÉ - DATASETS DE QUALITÉ")
print("="*70)

# GPU Setup
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("❌ GPU non disponible!")
    sys.exit(1)

# CONFIG
CONFIG = {
    'vocab_size': 50257,
    'embed_dim': 768,
    'num_heads': 12,
    'num_layers': 12,
    'max_seq_len': 1024,
    'dropout': 0.05,
    
    'batch_size': 12,
    'gradient_accumulation': 2,
    'num_epochs': 4,  # Total cumulé sur plusieurs sessions
    
    'learning_rate': 1e-5,
    'warmup_steps': 800,
    'max_grad_norm': 1.0,
    
    # DATASETS SIMPLIFIÉS (QUALITÉ)
    'alpaca_samples': None,      # Tous les samples valides
    'stanford_samples': None,    # Tous les samples valides
    'dolly_samples': None,       # Tous les samples valides
    'wizard_samples': None,     # Tous (~70K)
    'synthetic_samples': 280000,  
    
    # FILTRES CORRIGÉS
    'min_length': 10,            # 20 → 10 (permet réponses courtes)
    'max_length': 512,
    'min_assistant_tokens': 3,   # Min 3 tokens (évite vides)
    'max_assistant_tokens': 200, # 150 → 200
    'max_token_repetition': 3,   # Anti-répétition
    
    'patience': 3,
    'val_split': 0.05,           
    'min_val_loss_improvement': 0.001,
    
    'scheduler_type': 'cosine',
    'min_lr_ratio': 0.1,
    'weight_decay': 0.001,
}

print(f"\n⚙️  Configuration:")
print(f"  📊 Datasets: Alpaca + Stanford + Dolly + WizardLM + Synthétique")
print(f"  🎯 Focus: Instructions générales (pas de math/code)")
print(f"  🔄 Epochs: {CONFIG['num_epochs']} (stratégie multi-sessions)")
print(f"  ⏱️  Temps par session: ~4-5h (6 epochs)")
print(f"  🔁 Sessions prévues: 3 (6+6+6 epochs)")

# Tokenizer
print("\n🔤 Chargement tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer: {len(tokenizer)} tokens")

# Dataset Class
class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = item['instruction'].strip()
        input_text = item.get('input', '').strip()
        output = item['output'].strip()
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        response = f" {output}"
        
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        all_tokens = prompt_tokens + response_tokens
        all_tokens.append(self.tokenizer.eos_token_id)
        
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length]
        
        input_ids = torch.tensor(all_tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(all_tokens[1:], dtype=torch.long)
        
        mask = torch.ones_like(target_ids) * -100
        output_start = len(prompt_tokens)
        mask[output_start:] = target_ids[output_start:]
        
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.full((pad_length,), -100, dtype=torch.long)])
        
        return input_ids, mask

# FILTRES DE QUALITÉ
def is_valid_sample(instruction, output, tokenizer, config):
    """Filtre de qualité avec anti-répétition"""
    if not instruction or not output:
        return False
    
    if len(output) < 5 or len(instruction) < 3:
        return False
    
    # Vérifier longueur tokens
    output_tokens = tokenizer.encode(output)
    if len(output_tokens) < config['min_assistant_tokens']:
        return False
    if len(output_tokens) > config['max_assistant_tokens']:
        return False
    
    # Vérifier répétitions
    words = output.split()
    if len(words) > 5:
        # Ratio mots uniques
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:  # Au moins 40% de mots uniques
            return False
        
        # Vérifier répétitions consécutives
        for i in range(len(words) - config['max_token_repetition']):
            window = words[i:i+config['max_token_repetition']]
            if len(set(window)) == 1:
                return False
    
    # Vérifier longueur totale
    full = f"Instruction: {instruction}\nResponse: {output}"
    tokens = tokenizer.encode(full)
    
    return config['min_length'] < len(tokens) < config['max_length']

# CHARGEMENT DES DONNÉES
print("\n" + "="*70)
print("📥 CHARGEMENT DES DATASETS (QUALITÉ)")
print("="*70)

os.makedirs("data", exist_ok=True)
all_data = []

# 1. ALPACA COMPLET
print("\n📚 [1/5] Chargement Alpaca (COMPLET)...")
cache_alpaca = "data/alpaca_quality_full.pt"

if os.path.exists(cache_alpaca):
    print(f"✓ Cache trouvé")
    alpaca_data = torch.load(cache_alpaca)['data']
else:
    print("📥 Téléchargement...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_data = []
    
    for item in tqdm(alpaca, desc="Processing Alpaca"):
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output = item.get('output', '').strip()
        
        if not is_valid_sample(instruction, output, tokenizer, CONFIG):
            continue
        
        alpaca_data.append({
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'source': 'alpaca'
        })
    
    random.seed(42)
    random.shuffle(alpaca_data)
    torch.save({'data': alpaca_data}, cache_alpaca)

all_data.extend(alpaca_data)
print(f"✓ Alpaca: {len(alpaca_data):,} samples")

# 2. STANFORD ALPACA CLEANED
print("\n📚 [2/5] Chargement Stanford (COMPLET)...")
cache_stanford = "data/stanford_quality_full.pt"

if os.path.exists(cache_stanford):
    stanford_data = torch.load(cache_stanford)['data']
else:
    print("📥 Téléchargement...")
    try:
        stanford = load_dataset("yahma/alpaca-cleaned", split="train")
        stanford_data = []
        
        for item in tqdm(stanford, desc="Processing Stanford"):
            instruction = item.get('instruction', '').strip()
            input_text = item.get('input', '').strip()
            output = item.get('output', '').strip()
            
            if not is_valid_sample(instruction, output, tokenizer, CONFIG):
                continue
            
            stanford_data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output,
                'source': 'stanford'
            })
        
        random.shuffle(stanford_data)
        torch.save({'data': stanford_data}, cache_stanford)
        
    except Exception as e:
        print(f"⚠️  Erreur: {e}")
        stanford_data = []

all_data.extend(stanford_data)
print(f"✓ Stanford: {len(stanford_data):,} samples")

# 3. DOLLY COMPLET
print("\n📚 [3/5] Chargement Dolly (COMPLET)...")
cache_dolly = "data/dolly_quality_full.pt"

if os.path.exists(cache_dolly):
    dolly_data = torch.load(cache_dolly)['data']
else:
    print("📥 Téléchargement...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_data = []
    
    for item in tqdm(dolly, desc="Processing Dolly"):
        instruction = item.get('instruction', '').strip()
        context = item.get('context', '').strip()
        response = item.get('response', '').strip()
        
        if not is_valid_sample(instruction, response, tokenizer, CONFIG):
            continue
        
        dolly_data.append({
            'instruction': instruction,
            'input': context,
            'output': response,
            'source': 'dolly'
        })
    
    random.shuffle(dolly_data)
    torch.save({'data': dolly_data}, cache_dolly)

all_data.extend(dolly_data)
print(f"✓ Dolly: {len(dolly_data):,} samples")

# 4. WIZARDLM COMPLET
print("\n📚 [4/5] Chargement WizardLM (COMPLET)...")
cache_wizard = "data/wizard_quality_full.pt"

if os.path.exists(cache_wizard):
    wizard_data = torch.load(cache_wizard)['data']
else:
    print("📥 Téléchargement...")
    try:
        wizard = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
        wizard_data = []
        
        for item in tqdm(wizard, desc="Processing WizardLM"):
            instruction = item.get('instruction', '').strip()
            output = item.get('output', '').strip()
            
            if not is_valid_sample(instruction, output, tokenizer, CONFIG):
                continue
            
            wizard_data.append({
                'instruction': instruction,
                'input': '',
                'output': output,
                'source': 'wizardlm'
            })
        
        random.shuffle(wizard_data)
        torch.save({'data': wizard_data}, cache_wizard)
        
    except Exception as e:
        print(f"⚠️  Erreur: {e}")
        wizard_data = []

all_data.extend(wizard_data)
print(f"✓ WizardLM: {len(wizard_data):,} samples")

# 5. SYNTHÉTIQUE 100K
print("\n📚 [5/5] Chargement Synthétique 100K...")
synthetic_file = "synthetic_10k.jsonl"

if os.path.exists(synthetic_file):
    print(f"✓ Fichier trouvé")
    synthetic_data = []
    
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= CONFIG['synthetic_samples']:
                break
            try:
                obj = json.loads(line)
                # Convertir au format Alpaca
                user_text = obj['user'].strip()
                assistant_text = obj['assistant'].strip()
                
                # Vérifier validité
                if not is_valid_sample(user_text, assistant_text, tokenizer, CONFIG):
                    continue
                
                synthetic_data.append({
                    'instruction': user_text,
                    'input': '',
                    'output': assistant_text,
                    'source': 'synthetic'
                })
            except Exception as e:
                continue
    
    print(f"✓ {len(synthetic_data)} samples chargés")
else:
    print(f"⚠️  Fichier introuvable: {synthetic_file}")
    print("💡 Générez-le d'abord avec votre script Python")
    synthetic_data = []

all_data.extend(synthetic_data)
if synthetic_data:
    print(f"✓ Synthétique: {len(synthetic_data):,} samples ajoutés")

# STATISTIQUES
print("\n" + "="*70)
print("📊 DATASET FINAL (QUALITÉ OPTIMISÉE)")
print("="*70)

print(f"\n🎯 Composition:")
print(f"  Alpaca:      {len(alpaca_data):>7,} ({len(alpaca_data)/len(all_data)*100:>5.1f}%)")
print(f"  Stanford:    {len(stanford_data):>7,} ({len(stanford_data)/len(all_data)*100:>5.1f}%)")
print(f"  Dolly:       {len(dolly_data):>7,} ({len(dolly_data)/len(all_data)*100:>5.1f}%)")
print(f"  WizardLM:    {len(wizard_data):>7,} ({len(wizard_data)/len(all_data)*100:>5.1f}%)")
print(f"  Synthétique: {len(synthetic_data):>7,} ({len(synthetic_data)/len(all_data)*100:>5.1f}%)")
print(f"  {'─'*45}")
print(f"  TOTAL:       {len(all_data):>7,} samples")

# Analyse par longueur de réponse
short = sum(1 for d in all_data if len(tokenizer.encode(d['output'])) <= 20)
medium = sum(1 for d in all_data if 20 < len(tokenizer.encode(d['output'])) <= 80)
long_resp = sum(1 for d in all_data if len(tokenizer.encode(d['output'])) > 80)

print(f"\n📏 Distribution par longueur:")
print(f"  Courtes (≤20 tokens):  {short:>7,} ({short/len(all_data)*100:>5.1f}%)")
print(f"  Moyennes (20-80):      {medium:>7,} ({medium/len(all_data)*100:>5.1f}%)")
print(f"  Longues (>80):         {long_resp:>7,} ({long_resp/len(all_data)*100:>5.1f}%)")

# Split & Dataloader
print("\n📊 Split train/validation...")
random.shuffle(all_data)
split_idx = int(len(all_data) * (1 - CONFIG['val_split']))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

print(f"✓ Train: {len(train_data):,} samples")
print(f"✓ Val: {len(val_data):,} samples")

train_dataset = AlpacaDataset(train_data, tokenizer, CONFIG['max_length'])
val_dataset = AlpacaDataset(val_data, tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

print(f"✓ Train batches: {len(train_loader):,}")

# MODÈLE
print("\n🤖 Chargement modèle...")
checkpoint = torch.load("./checkpoints/hessgpt_gpt2_final.pt", map_location=device)
model = HessGPT(
    vocab_size=CONFIG['vocab_size'], embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'], num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len'], dropout=CONFIG['dropout']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Paramètres: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Optimizer & Scheduler
total_steps = (len(train_loader) * CONFIG['num_epochs']) // CONFIG['gradient_accumulation']
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'], fused=True)

def lr_lambda(step):
    if step < CONFIG['warmup_steps']:
        return step / CONFIG['warmup_steps']
    progress = (step - CONFIG['warmup_steps']) / max(total_steps - CONFIG['warmup_steps'], 1)
    return CONFIG['min_lr_ratio'] + (1.0 - CONFIG['min_lr_ratio']) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            if not torch.isnan(loss):
                total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)

# SYSTÈME DE REPRISE DE TRAINING
print("\n" + "="*70)
print("🔄 VÉRIFICATION CHECKPOINT DE REPRISE")
print("="*70)

os.makedirs("checkpoints/quality", exist_ok=True)

# Initialiser le scaler AVANT de charger le checkpoint
scaler = torch.amp.GradScaler('cuda')

# Checkpoint de reprise automatique
resume_checkpoint = "./checkpoints/quality/hessgpt_sft_RESUME.pt"
start_epoch = 0
total_samples_seen = 0
best_val_loss = float('inf')
patience_counter = 0
training_history = []

if os.path.exists(resume_checkpoint):
    print(f"\n🔍 Checkpoint de reprise trouvé!")
    resume_data = torch.load(resume_checkpoint, map_location=device)
    
    # Restaurer le modèle
    model.load_state_dict(resume_data['model_state_dict'])
    print(f"✓ Modèle restauré")
    
    # Restaurer optimizer & scheduler
    optimizer.load_state_dict(resume_data['optimizer_state_dict'])
    scheduler.load_state_dict(resume_data['scheduler_state_dict'])
    print(f"✓ Optimizer & Scheduler restaurés")
    
    # Restaurer scaler (AMP)
    scaler.load_state_dict(resume_data['scaler_state_dict'])
    print(f"✓ Scaler AMP restauré")
    
    # Restaurer infos training
    start_epoch = resume_data['epoch']
    total_samples_seen = resume_data.get('total_samples_seen', 0)
    best_val_loss = resume_data.get('best_val_loss', float('inf'))
    patience_counter = resume_data.get('patience_counter', 0)
    training_history = resume_data.get('training_history', [])
    
    print(f"\n📊 État du training:")
    print(f"  • Epoch: {start_epoch}/{CONFIG['num_epochs']}")
    print(f"  • Samples vus: {total_samples_seen:,}")
    print(f"  • Best Val Loss: {best_val_loss:.4f}")
    print(f"  • Patience: {patience_counter}/{CONFIG['patience']}")
    print(f"\n✅ Reprise du training à partir de l'epoch {start_epoch + 1}")
else:
    print(f"\n💡 Aucun checkpoint de reprise trouvé")
    print(f"✅ Démarrage d'un nouveau training")

# TRAINING
print("\n" + "="*70)
print("🚀 TRAINING - VERSION QUALITÉ")
print("="*70)

model.train()
scaler = torch.amp.GradScaler('cuda')
if os.path.exists(resume_checkpoint):
    scaler.load_state_dict(torch.load(resume_checkpoint, map_location=device)['scaler_state_dict'])

start_time = time.time()

for epoch in range(start_epoch, CONFIG['num_epochs']):
    epoch_loss = 0
    epoch_samples = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        batch_samples = x.size(0)
        epoch_samples += batch_samples
        total_samples_seen += batch_samples
        
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100) / CONFIG['gradient_accumulation']
        
        if not torch.isnan(loss):
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG['gradient_accumulation'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            epoch_loss += loss.item() * CONFIG['gradient_accumulation']
            
            # Affichage enrichi
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}',
                'lr': f'{current_lr:.2e}',
                'samples': f'{total_samples_seen:,}'
            })
    
    val_loss = validate(model, val_loader)
    elapsed = (time.time() - start_time) / 3600
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Sauvegarder dans l'historique
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'learning_rate': scheduler.get_last_lr()[0],
        'samples_seen': total_samples_seen,
        'time_hours': elapsed,
    })
    
    print(f"\n✓ Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"  • Train Loss: {avg_train_loss:.4f}")
    print(f"  • Val Loss: {val_loss:.4f}")
    print(f"  • Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
    print(f"  • Samples vus (epoch): {epoch_samples:,}")
    print(f"  • Samples vus (total): {total_samples_seen:,}")
    print(f"  • Temps écoulé: {elapsed:.2f}h")
    
    improvement = best_val_loss - val_loss
    
    # SAUVEGARDE CHECKPOINT DE REPRISE (à chaque epoch)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': CONFIG,
        'val_loss': val_loss,
        'train_loss': avg_train_loss,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'total_samples_seen': total_samples_seen,
        'training_history': training_history,
        'composition': {
            'alpaca': len(alpaca_data),
            'stanford': len(stanford_data),
            'dolly': len(dolly_data),
            'wizardlm': len(wizard_data),
            'synthetic': len(synthetic_data),
        }
    }, resume_checkpoint)
    print(f"  💾 Checkpoint de reprise sauvegardé")
    
    if improvement >= CONFIG['min_val_loss_improvement']:
        best_val_loss = val_loss
        patience_counter = 0
        
        # SAUVEGARDE MEILLEUR MODÈLE
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'val_loss': val_loss,
            'train_loss': avg_train_loss,
            'total_samples_seen': total_samples_seen,
            'training_history': training_history,
            'composition': {
                'alpaca': len(alpaca_data),
                'stanford': len(stanford_data),
                'dolly': len(dolly_data),
                'wizardlm': len(wizard_data),
                'synthetic': len(synthetic_data),
            }
        }, './checkpoints/quality/hessgpt_sft_quality_BEST.pt')
        
        print(f"  🏆 MEILLEUR MODÈLE! (Val: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  ⚠️  Patience: {patience_counter}/{CONFIG['patience']}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 EARLY STOPPING (Epoch {epoch+1}/{CONFIG['num_epochs']})")
            break

# RÉSUMÉ FINAL
total_time = (time.time() - start_time) / 3600

print("\n" + "="*70)
print("✅ TRAINING TERMINÉ!")
print("="*70)

print(f"\n📊 STATISTIQUES FINALES:")
print(f"  🏆 Best Val Loss: {best_val_loss:.4f}")
print(f"  ⏱️  Temps total: {total_time:.2f}h")
print(f"  📁 Total samples: {len(all_data):,}")
print(f"  👁️  Samples vus: {total_samples_seen:,}")
print(f"  🔄 Epochs complétés: {len(training_history)}/{CONFIG['num_epochs']}")

print(f"\n📊 Composition du dataset:")
print(f"  • Alpaca: {len(alpaca_data):,}")
print(f"  • Stanford: {len(stanford_data):,}")
print(f"  • Dolly: {len(dolly_data):,}")
print(f"  • WizardLM: {len(wizard_data):,}")
print(f"  • Synthétique: {len(synthetic_data):,}")

print(f"\n📈 Historique de training:")
for i, hist in enumerate(training_history[-5:], start=max(1, len(training_history)-4)):
    print(f"  Epoch {hist['epoch']:2d} | Train: {hist['train_loss']:.4f} | Val: {hist['val_loss']:.4f} | LR: {hist['learning_rate']:.2e}")

print(f"\n💾 Fichiers sauvegardés:")
print(f"  • Meilleur modèle: ./checkpoints/quality/hessgpt_sft_quality_BEST.pt")
print(f"  • Checkpoint reprise: ./checkpoints/quality/hessgpt_sft_RESUME.pt")

# Sauvegarder historique en JSON
history_file = './checkpoints/quality/training_history.json'
with open(history_file, 'w') as f:
    json.dump({
        'config': {k: v for k, v in CONFIG.items() if not callable(v)},
        'final_stats': {
            'best_val_loss': best_val_loss,
            'total_time_hours': total_time,
            'total_samples': len(all_data),
            'samples_seen': total_samples_seen,
            'epochs_completed': len(training_history),
        },
        'composition': {
            'alpaca': len(alpaca_data),
            'stanford': len(stanford_data),
            'dolly': len(dolly_data),
            'wizardlm': len(wizard_data),
            'synthetic': len(synthetic_data),
        },
        'training_history': training_history,
    }, f, indent=2)
print(f"  • Historique JSON: {history_file}")

print("\n" + "="*70)
print("🎯 PROCHAINES ÉTAPES:")
print("="*70)
print("\n1. 🧪 Tester le modèle:")
print("   python test_hessgpt_sft.py")
print("\n2. 🎮 Mode interactif:")
print("   python hessgpt_inference.py")
print("\n3. 🔄 Reprendre le training (si interrompu):")
print("   Relancez simplement ce script, il reprendra automatiquement!")
print("\n" + "="*70)