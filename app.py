"""
🚀 HessGPT Web Interface - Flask Server
✅ Support multi-modèles (124M, 50M, 20M)
✅ API REST + Interface HTML moderne
✅ Prêt pour production
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import sys
import os

# Importer ton modèle
sys.path.append('./Core/Model')
from HessGpt import HessGPT

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION DES MODÈLES
# ============================================

MODELS_CONFIG = {
    '124M': {
        'path': './Models/124M/Hessgpt_Final_SFT.pt',
        'config': {
            'vocab_size': 50257,
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12,
            'max_seq_len': 1024,
            'dropout': 0.05,
        },
        'description': 'Modèle large - V3'
    },
    '50M': {
        'path': './Models/50M/Hessgpt_Final_SFT.pt',
        'config': {
            'vocab_size': 50257,
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 8,
            'max_seq_len': 1024,
            'dropout': 0.05,
        },
        'description': 'Modèle moyen - V2'
    },
    '20M': {
        'path': './Models/20M/Hessgpt_Final_SFT.pt',
        'config': {
            'vocab_size': 50257,
            'embed_dim': 384,
            'num_heads': 6,
            'num_layers': 6,
            'max_seq_len': 1024,
            'dropout': 0.05,
        },
        'description': 'Modèle léger - V1'
    }
}

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Stockage des modèles chargés
loaded_models = {}
current_model_name = None

# ============================================
# INITIALISATION
# ============================================

print("="*60)
print("🚀 DÉMARRAGE SERVEUR HessGPT MULTI-MODÈLES")
print("="*60)
print(f"✅ Device: {DEVICE}")

# Tokenizer (commun à tous les modèles)
tokenizer = GPT2Tokenizer.from_pretrained("./Core/Tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# Vérifier quels modèles sont disponibles
available_models = []
for model_name, model_info in MODELS_CONFIG.items():
    if os.path.exists(model_info['path']):
        available_models.append(model_name)
        print(f"✓ Modèle {model_name} trouvé: {model_info['path']}")
    else:
        print(f"✗ Modèle {model_name} absent: {model_info['path']}")

if not available_models:
    print("❌ ERREUR: Aucun modèle trouvé!")
    print("📁 Vérifiez la structure: ./Models/[124M|50M|20M]/Hessgpt_Final_SFT.pt")
    sys.exit(1)

print(f"\n✅ {len(available_models)} modèle(s) disponible(s): {', '.join(available_models)}")
print("="*60)

# ============================================
# GESTION DES MODÈLES
# ============================================

def load_model(model_name):
    """Charge un modèle spécifique"""
    global current_model_name
    
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Modèle {model_name} inconnu")
    
    if model_name not in available_models:
        raise ValueError(f"Modèle {model_name} non disponible")
    
    # Si déjà chargé, le retourner
    if model_name in loaded_models:
        current_model_name = model_name
        return loaded_models[model_name]
    
    print(f"\n⏳ Chargement du modèle {model_name}...")
    
    model_info = MODELS_CONFIG[model_name]
    
    # Charger le checkpoint
    checkpoint = torch.load(model_info['path'], map_location=DEVICE)
    
    # Créer et initialiser le modèle
    model = HessGPT(**model_info['config']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Stocker
    loaded_models[model_name] = {
        'model': model,
        'config': model_info['config'],
        'checkpoint': checkpoint
    }
    current_model_name = model_name
    
    print(f"✅ Modèle {model_name} chargé!")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))}")
    
    return loaded_models[model_name]

# Charger le premier modèle disponible par défaut
default_model = available_models[0]
load_model(default_model)

# ============================================
# FONCTION DE GÉNÉRATION
# ============================================

def generate_response(prompt, model_name=None, max_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    """
    Génère une réponse avec le modèle HessGPT
    """
    if model_name is None:
        model_name = current_model_name
    
    # Charger le modèle si nécessaire
    model_data = load_model(model_name)
    model = model_data['model']
    config = model_data['config']
    
    model.eval()
    
    # Formater le prompt (style Alpaca)
    formatted_prompt = f"Instruction: {prompt}\nResponse:"
    
    # Tokenization
    tokens = tokenizer.encode(formatted_prompt, return_tensors='pt').to(DEVICE)
    generated = tokens[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            
            if input_ids.size(1) > config['max_seq_len']:
                input_ids = input_ids[:, -config['max_seq_len']:]
            
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Température
            next_token_logits = next_token_logits / temperature
            
            # Anti-répétition
            for token in set(generated[-50:]):
                next_token_logits[token] /= 1.2
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
    
    # Décoder
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    
    if "Response:" in full_text:
        response = full_text.split("Response:")[-1].strip()
    else:
        response = full_text[len(formatted_prompt):].strip()
    
    return response

# ============================================
# ROUTES FLASK
# ============================================

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    """Retourne la liste des modèles disponibles"""
    models_list = []
    for model_name in available_models:
        model_info = MODELS_CONFIG[model_name]
        models_list.append({
            'name': model_name,
            'description': model_info['description'],
            'active': model_name == current_model_name
        })
    
    return jsonify({
        'models': models_list,
        'current': current_model_name
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Change de modèle actif"""
    try:
        data = request.get_json()
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({'error': 'Nom de modèle manquant', 'success': False}), 400
        
        load_model(model_name)
        
        return jsonify({
            'success': True,
            'model': model_name,
            'message': f'Modèle {model_name} activé'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """API de génération"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt manquant', 'success': False}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Prompt vide', 'success': False}), 400
        
        # Paramètres
        model_name = data.get('model', current_model_name)
        max_tokens = min(int(data.get('max_tokens', 100)), 500)
        temperature = max(0.1, min(float(data.get('temperature', 0.7)), 1.0))
        
        # Génération
        response = generate_response(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return jsonify({
            'response': response,
            'success': True,
            'model': current_model_name,
            'params': {
                'max_tokens': max_tokens,
                'temperature': temperature
            }
        })
    
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return jsonify({'error': f'Erreur serveur: {str(e)}', 'success': False}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Efface l'historique"""
    return jsonify({'success': True, 'message': 'Conversation effacée'})

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'current_model': current_model_name,
        'available_models': available_models
    })

@app.route('/info', methods=['GET'])
def info():
    """Informations sur le modèle actuel"""
    if current_model_name not in loaded_models:
        return jsonify({'error': 'Aucun modèle chargé'}), 500
    
    model_data = loaded_models[current_model_name]
    checkpoint = model_data['checkpoint']
    
    return jsonify({
        'model': current_model_name,
        'description': MODELS_CONFIG[current_model_name]['description'],
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_loss': checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A')),
        'config': model_data['config'],
        'device': DEVICE,
        'samples_seen': checkpoint.get('total_samples_seen', 'N/A')
    })

# ============================================
# DÉMARRAGE SERVEUR
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Serveur démarré!")
    print("="*60)
    print("📍 Interface: http://localhost:5000")
    print("📍 API: http://localhost:5000/generate")
    print("📍 Modèles: http://localhost:5000/models")
    print("📍 Health: http://localhost:5000/health")
    print(f"📍 Modèle actif: {current_model_name}")
    print("="*60)
    print("\n⚠️  Utiliser CTRL+C pour arrêter\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )