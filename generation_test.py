"""
🧪 BENCHMARK HessGPT - TESTS SYNTHÉTIQUES UNIQUEMENT
✅ Questions alignées avec le dataset synthétique 30K
✅ Conversations naturelles courtes
✅ Scoring automatique
"""

import torch
import torch.nn.functional as F
import sys, time
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.append('./Core/Model')
from HessGpt import HessGPT

print("="*70)
print("🧪 BENCHMARK HessGPT - TESTS SYNTHÉTIQUES")
print("="*70)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✅ Device: {device}")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# CONFIG
CONFIG = {
    'vocab_size': 50257,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'max_seq_len': 1024,
    'dropout': 0.05,
}

# Chargement modèle SFT
print("\n🤖 Chargement modèle SFT...")

checkpoint_paths = [
    "./checkpoints/quality/hessgpt_sft_quality_BEST.pt",
    "./checkpoints/quality/hessgpt_sft_RESUME.pt"
]

checkpoint = None
for path in checkpoint_paths:
    try:
        checkpoint = torch.load(path, map_location=device)
        print(f"✓ Checkpoint chargé: {path.split('/')[-1]}")
        break
    except FileNotFoundError:
        continue

if checkpoint is None:
    print("❌ Aucun checkpoint trouvé!")
    sys.exit(1)

model = HessGPT(**CONFIG).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))
print(f"✓ Val Loss: {val_loss}")
print(f"✓ Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"✓ Samples vus: {checkpoint.get('total_samples_seen', 'N/A'):,}" if checkpoint.get('total_samples_seen') else "")

# FONCTION DE GÉNÉRATION
def generate_text(model, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = tokens[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            next_token_logits = next_token_logits / temperature
            
            # Anti-répétition
            for token in set(generated[-50:]):
                next_token_logits[token] /= 1.2
            
            # Top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
    
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    if "Response:" in full_text:
        return full_text.split("Response:")[-1].strip()
    return full_text[len(tokenizer.decode(tokens[0], skip_special_tokens=True)):].strip()

# TESTS BASÉS SUR LE DATASET SYNTHÉTIQUE
benchmark_tests = {
    "👋 SALUTATIONS": [
        {"q": "Hi there!", "expected_keywords": ["hello", "hi", "hey", "how can", "help"]},
        {"q": "Hello!", "expected_keywords": ["hi", "hello", "hey", "how can", "help"]},
        {"q": "Hey!", "expected_keywords": ["hey", "hi", "hello", "what's", "help"]},
        {"q": "Good morning!", "expected_keywords": ["good", "morning", "day", "wonderful", "hope"]},
        {"q": "Howdy!", "expected_keywords": ["howdy", "hi", "hello", "what can", "do for you"]},
        {"q": "Greetings!", "expected_keywords": ["greetings", "hi", "hello", "assist", "help"]},
    ],
    
    "💬 COMMENT ÇA VA": [
        {"q": "How are you doing?", "expected_keywords": ["great", "good", "doing", "well", "thanks"]},
        {"q": "How's it going?", "expected_keywords": ["well", "good", "great", "how about", "you"]},
        {"q": "What's up?", "expected_keywords": ["not much", "just", "chat", "you", "what about"]},
        {"q": "How are you?", "expected_keywords": ["good", "great", "fine", "well", "you"]},
        {"q": "How's everything?", "expected_keywords": ["good", "great", "all", "you"]},
    ],
    
    "🙏 REMERCIEMENTS": [
        {"q": "Thanks a lot!", "expected_keywords": ["welcome", "you're", "very"]},
        {"q": "Thank you!", "expected_keywords": ["welcome", "anytime", "pleasure", "glad"]},
        {"q": "I appreciate it.", "expected_keywords": ["happy", "help", "anytime", "glad"]},
        {"q": "Thanks!", "expected_keywords": ["welcome", "anytime", "glad"]},
        {"q": "Big thanks!", "expected_keywords": ["happy", "help", "welcome"]},
    ],
    
    "👋 AU REVOIR": [
        {"q": "Goodbye!", "expected_keywords": ["goodbye", "bye", "take care", "see you"]},
        {"q": "Bye!", "expected_keywords": ["bye", "stay", "safe", "goodbye", "see you"]},
        {"q": "See you later!", "expected_keywords": ["see you", "great", "day", "later"]},
        {"q": "Catch you later.", "expected_keywords": ["later", "take care", "bye"]},
        {"q": "Gotta run.", "expected_keywords": ["no problem", "talk", "later", "bye"]},
    ],
    
    "🤔 QUESTIONS SIMPLES": [
        {"q": "What's your name?", "expected_keywords": ["claude", "ai", "assistant"]},
        {"q": "Who are you?", "expected_keywords": ["claude", "ai", "assistant", "here", "assist"]},
        {"q": "Can you help me?", "expected_keywords": ["course", "of course", "what", "need", "help"]},
        {"q": "Are you busy?", "expected_keywords": ["never", "too busy", "chat", "not busy"]},
        {"q": "What can you do?", "expected_keywords": ["chat", "answer", "help", "questions", "tasks"]},
    ],
    
    "✅ CONFIRMATIONS": [
        {"q": "Okay.", "expected_keywords": ["sounds good", "good", "great"]},
        {"q": "Alright.", "expected_keywords": ["perfect", "what's next", "good"]},
        {"q": "Got it!", "expected_keywords": ["excellent", "great", "let me know", "need more"]},
        {"q": "Makes sense.", "expected_keywords": ["perfect", "good", "anything else"]},
        {"q": "I see.", "expected_keywords": ["good", "any", "questions", "else"]},
    ],
    
    "❓ DEMANDES DE CLARIFICATION": [
        {"q": "What?", "expected_keywords": ["what", "would", "like", "know", "help"]},
        {"q": "Huh?", "expected_keywords": ["sorry", "clarify", "let me"]},
        {"q": "Can you repeat that?", "expected_keywords": ["sure", "repeat", "what would"]},
        {"q": "I didn't catch that.", "expected_keywords": ["no worries", "let me", "clarify"]},
        {"q": "Say that again?", "expected_keywords": ["course", "which", "part"]},
    ],
    
    "😊 COMPLIMENTS": [
        {"q": "You're awesome!", "expected_keywords": ["thanks", "kind", "you", "appreciate"]},
        {"q": "Great job!", "expected_keywords": ["thank", "you", "much", "appreciate"]},
        {"q": "You're helpful.", "expected_keywords": ["thank", "you", "means", "lot"]},
        {"q": "You rock!", "expected_keywords": ["thanks", "you", "great", "too"]},
        {"q": "Well done!", "expected_keywords": ["thanks", "appreciate"]},
    ],
    
    "🎭 ÉMOTIONS": [
        {"q": "I'm happy!", "expected_keywords": ["wonderful", "great", "what", "made", "day"]},
        {"q": "I'm sad.", "expected_keywords": ["tough", "sorry", "here", "talk", "chat"]},
        {"q": "I'm tired.", "expected_keywords": ["maybe", "time", "rest"]},
        {"q": "I'm bored.", "expected_keywords": ["let's", "find", "interesting", "talk"]},
        {"q": "I'm excited!", "expected_keywords": ["great", "what", "excited", "about"]},
    ],
    
    "🗣️ PHRASES COURTES": [
        {"q": "Yes?", "expected_keywords": ["how", "can", "help", "assist"]},
        {"q": "No?", "expected_keywords": ["alright", "anything", "else"]},
        {"q": "Maybe.", "expected_keywords": ["fair", "enough", "take", "time"]},
        {"q": "Sure.", "expected_keywords": ["sounds", "good", "great"]},
        {"q": "Cool.", "expected_keywords": ["glad", "you", "think"]},
    ],
}

# FONCTION DE SCORING ADAPTÉE AU SYNTHÉTIQUE
def score_response(response, test_case):
    """Score basé sur les patterns du dataset synthétique"""
    score = 0
    response_lower = response.lower()
    words = response.split()
    
    # 1. Longueur appropriée (réponses courtes du synthétique: 3-20 mots)
    if 3 <= len(words) <= 25:
        score += 30
    elif len(words) > 25:
        score += 10  # Pénalité légère si trop long
    
    # 2. Pas de répétition
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        score += int(unique_ratio * 20)
    
    # 3. Pas de tokens bizarres ou de format Alpaca
    if not any(c in response for c in ['�', '\\x', '<|', 'Instruction:', 'Input:']):
        score += 20
    
    # 4. Mots-clés attendus (spécifique au synthétique)
    if 'expected_keywords' in test_case:
        keywords_found = sum(1 for kw in test_case['expected_keywords'] 
                           if kw.lower() in response_lower)
        if keywords_found > 0:
            score += int((keywords_found / len(test_case['expected_keywords'])) * 30)
    
    return min(score, 100)

# EXÉCUTION DU BENCHMARK
print("\n" + "="*70)
print("🧪 DÉBUT DU BENCHMARK - TESTS SYNTHÉTIQUES")
print("="*70)

all_results = {}
total_score = 0
total_tests = 0

for category, tests in benchmark_tests.items():
    print(f"\n{'='*70}")
    print(category)
    print('='*70)
    
    category_scores = []
    
    for i, test in enumerate(tests, 1):
        question = test['q']
        
        # Format simple comme dans le synthétique
        prompt = f"Instruction: {question}\nResponse:"
        
        # Générer
        print(f"\n[{i}/{len(tests)}] 📝 Question: {question}")
        
        start_time = time.time()
        response = generate_text(model, prompt, max_length=80)
        gen_time = time.time() - start_time
        
        # Scorer
        score = score_response(response, test)
        category_scores.append(score)
        total_score += score
        total_tests += 1
        
        # Afficher
        print(f"    🤖 Réponse ({gen_time:.2f}s): {response}")
        print(f"    📊 Score: {score}/100 {'✅' if score >= 70 else '⚠️' if score >= 50 else '❌'}")
    
    # Stats catégorie
    avg_score = sum(category_scores) / len(category_scores)
    all_results[category] = {
        'scores': category_scores,
        'avg': avg_score,
        'count': len(tests)
    }
    
    print(f"\n{'─'*70}")
    print(f"📈 Moyenne catégorie: {avg_score:.1f}/100")

# RÉSUMÉ FINAL
print("\n" + "="*70)
print("📊 RÉSUMÉ FINAL DU BENCHMARK")
print("="*70)

print(f"\n🎯 Résultats par catégorie:\n")
for category, results in all_results.items():
    avg = results['avg']
    emoji = "🔥" if avg >= 80 else "✅" if avg >= 70 else "⚠️" if avg >= 60 else "❌"
    print(f"  {emoji} {category}")
    print(f"     Score moyen: {avg:.1f}/100 ({results['count']} tests)")

# Score global
global_avg = total_score / total_tests
print(f"\n{'='*70}")
print(f"🏆 SCORE GLOBAL: {global_avg:.1f}/100")
print(f"{'='*70}")

# Interprétation
if global_avg >= 80:
    print("\n🔥 EXCELLENT! Le modèle maîtrise les conversations synthétiques.")
elif global_avg >= 70:
    print("\n✅ TRÈS BON! Le modèle gère bien les conversations courtes.")
elif global_avg >= 60:
    print("\n⚠️  BON! Le modèle comprend mais manque de naturel.")
else:
    print("\n❌ MOYEN. Le modèle a besoin de plus d'entraînement.")

print(f"\n📊 Détails:")
print(f"  • Tests totaux: {total_tests}")
print(f"  • Dataset: Synthétique 30K uniquement")
print(f"  • Val Loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))}")
print(f"  • Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  • Samples vus: {checkpoint.get('total_samples_seen', 'N/A'):,}" if checkpoint.get('total_samples_seen') else "")

print("\n" + "="*70)
print("✅ BENCHMARK TERMINÉ")
print("="*70)