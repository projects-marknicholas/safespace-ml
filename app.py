# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import traceback

# Configuration
class Config:
    BERT_MODEL = 'bert-base-uncased'
    MAX_LENGTH = 256
    HIDDEN_SIZE = 768
    NUM_OFFENSE_CLASSES = 5
    NUM_SEVERITY_CLASSES = 4  # 0=Light, 1=Less Grave, 2=Grave, 3=None
    RNN_HIDDEN_SIZE = 256
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.3
    BIDIRECTIONAL = True
    DROPOUT_RATE = 0.4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Model Architecture
class SkipRNNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = torch.nn.LayerNorm(output_size)
        if input_size != output_size:
            self.proj = torch.nn.Linear(input_size, output_size)
        else:
            self.proj = torch.nn.Identity()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        x_proj = self.proj(x)
        skip_out = self.layer_norm(self.dropout(rnn_out + x_proj))
        return skip_out

class AdvancedBERTSkipRNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        
        self.skip_rnn1 = SkipRNNLayer(
            config.HIDDEN_SIZE,
            config.RNN_HIDDEN_SIZE,
            config.RNN_NUM_LAYERS,
            config.RNN_DROPOUT,
            config.BIDIRECTIONAL
        )
        
        self.skip_rnn2 = SkipRNNLayer(
            config.RNN_HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1),
            config.RNN_HIDDEN_SIZE // 2,
            max(1, config.RNN_NUM_LAYERS // 2),
            config.RNN_DROPOUT,
            config.BIDIRECTIONAL
        )
        
        rnn_out_size = (config.RNN_HIDDEN_SIZE // 2) * (2 if config.BIDIRECTIONAL else 1)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=rnn_out_size,
            num_heads=4,
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )
        
        self.dropout = torch.nn.Dropout(config.DROPOUT_RATE)
        
        # Two classification heads
        self.offense_classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_out_size * 2, config.RNN_HIDDEN_SIZE // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.DROPOUT_RATE),
            torch.nn.Linear(config.RNN_HIDDEN_SIZE // 4, config.NUM_OFFENSE_CLASSES)
        )
        self.severity_classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_out_size * 2, config.RNN_HIDDEN_SIZE // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.DROPOUT_RATE),
            torch.nn.Linear(config.RNN_HIDDEN_SIZE // 4, config.NUM_SEVERITY_CLASSES)
        )
    
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_out.last_hidden_state
        
        rnn1 = self.skip_rnn1(seq_out)
        rnn2 = self.skip_rnn2(rnn1)
        
        attn_out, _ = self.attention(rnn2, rnn2, rnn2)
        
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        
        logits_offense = self.offense_classifier(self.dropout(combined))
        logits_severity = self.severity_classifier(self.dropout(combined))
        return logits_offense, logits_severity

# Load Model
print("Loading tokenizer and model...")
try:
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
    model = AdvancedBERTSkipRNN(config)
    
    # Try different checkpoint formats
    checkpoint_files = ['harassment_model_full.pt', 'best_model.pt']
    loaded = False
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(
                checkpoint_file, 
                map_location=config.DEVICE,
                weights_only=False  # I-set sa False para gumana
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {checkpoint_file} (full checkpoint format)")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model from {checkpoint_file} (state dict format)")
            
            loaded = True
            break
        except FileNotFoundError:
            print(f"File {checkpoint_file} not found, trying next...")
        except Exception as e:
            print(f"Error loading {checkpoint_file}: {e}")
    
    if not loaded:
        raise Exception("No valid checkpoint file found")
    
    model.to(config.DEVICE)
    model.eval()
    print(f"Model loaded successfully on {config.DEVICE}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    raise

# Labels (from training)
OFFENSE_LABELS = ['Physical Harassment', 'Verbal Harassment', 'Non-Verbal Harassment', 
                  'Not Harassment', 'Cyber Sexual Harassment']
SEVERITY_LABELS = ['Light', 'Less Grave', 'Grave', 'None']  # 0=Light, 1=Less Grave, 2=Grave, 3=None

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing description field'}), 400
        
        description = data['description']
        print(f"Processing: {description}")
        
        # Tokenize input
        inputs = tokenizer(
            description,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            offense_logits, severity_logits = model(
                inputs['input_ids'].to(config.DEVICE), 
                inputs['attention_mask'].to(config.DEVICE)
            )
            
            # Get probabilities
            offense_probs = torch.softmax(offense_logits, dim=1).cpu().numpy()[0]
            severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]
            
            # Get predictions
            offense_idx = np.argmax(offense_probs)
            severity_idx = np.argmax(severity_probs)
            
            # Debug output
            print(f"Offense probs: {dict(zip(OFFENSE_LABELS, offense_probs))}")
            print(f"Severity probs: {dict(zip(SEVERITY_LABELS, severity_probs))}")
            
            # If offense is "Not Harassment", severity should be "None" (index 3)
            # But the model might still predict a severity, so we can optionally override
            if OFFENSE_LABELS[offense_idx] == 'Not Harassment' and severity_idx != 3:
                print(f"Warning: Not Harassment detected but severity predicted as {SEVERITY_LABELS[severity_idx]}. Setting to None.")
                severity_idx = 3
                severity_probs = [0.0, 0.0, 0.0, 1.0]
        
        # Return response
        return jsonify({
            'offense': {
                'label': OFFENSE_LABELS[offense_idx],
                'confidence': float(offense_probs[offense_idx]),
                'probabilities': {label: float(offense_probs[i]) for i, label in enumerate(OFFENSE_LABELS)}
            },
            'severity': {
                'label': SEVERITY_LABELS[severity_idx],
                'confidence': float(severity_probs[severity_idx]),
                'probabilities': {label: float(severity_probs[i]) for i, label in enumerate(SEVERITY_LABELS)}
            }
        })
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        
        if not data or 'descriptions' not in data:
            return jsonify({'error': 'Missing descriptions field'}), 400
        
        descriptions = data['descriptions']
        if not isinstance(descriptions, list):
            return jsonify({'error': 'descriptions must be a list'}), 400
        
        results = []
        for description in descriptions:
            # Tokenize input
            inputs = tokenizer(
                description,
                max_length=config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Make prediction
            with torch.no_grad():
                offense_logits, severity_logits = model(
                    inputs['input_ids'].to(config.DEVICE), 
                    inputs['attention_mask'].to(config.DEVICE)
                )
                
                offense_probs = torch.softmax(offense_logits, dim=1).cpu().numpy()[0]
                severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]
                
                offense_idx = np.argmax(offense_probs)
                severity_idx = np.argmax(severity_probs)
                
                # Override severity for Not Harassment
                if OFFENSE_LABELS[offense_idx] == 'Not Harassment':
                    severity_idx = 3
                    severity_probs = [0.0, 0.0, 0.0, 1.0]
            
            results.append({
                'description': description,
                'offense': {
                    'label': OFFENSE_LABELS[offense_idx],
                    'confidence': float(offense_probs[offense_idx])
                },
                'severity': {
                    'label': SEVERITY_LABELS[severity_idx],
                    'confidence': float(severity_probs[severity_idx])
                }
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(config.DEVICE),
        'model_info': {
            'offense_classes': OFFENSE_LABELS,
            'severity_classes': SEVERITY_LABELS,
            'max_length': config.MAX_LENGTH
        }
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Harassment Detection Microservice',
        'version': '2.0',
        'endpoints': {
            '/predict': 'POST - Detect harassment type and severity from text',
            '/predict-batch': 'POST - Batch detection for multiple texts',
            '/health': 'GET - Check service health'
        },
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)  # Set debug=True for more detailed errors
