import sys
import os
import json
import traceback
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from torch.cuda import amp
from tqdm import tqdm

# Fix warning filter error
warnings.filterwarnings("ignore")

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar,
    QGroupBox, QFormLayout, QLineEdit, QSplitter, QMessageBox,
    QStatusBar, QGridLayout, QCheckBox, QStackedWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon,QPixmap,QPainter,QPen

# Add model training modules
# sys.path.append('./task/')
from task.Dataset.dataset import PPISDataset
from torch_geometric.data import DataLoader
from task.model_block.PPIsmodels import KD_EGNN_edge

# Add visualization components
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# from main import MainMenu

# Configuration class
class TrainingConfig:
    def __init__(self):
        self.infeature_size = 1152
        self.outfeature_size = 512
        self.seq_dim = 1152
        self.node_dim = 1152
        self.out_dim = 512
        self.nhidden_eg = 128
        self.edge_feature_size = 290
        self.n_eglayer = 4
        self.nclass = 2
        self.hidden_dim = 512
        self.batch_size = 1
        self.lr = 0.0001
        self.epochs = 50
        self.seed = 2025
        self.patience = 10
        self.contrastive_weight = 0.5
        self.kd_temperature = 3.0
        self.use_amp = True

config = TrainingConfig()

# Loss function
def CrossEntropy(outputs, targets, temperature=config.kd_temperature):
    log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
    softmax_targets = F.softmax(targets / temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

# GPU utility function
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training preparation functions
def prepare_cross_validation(config, esm_path, pdb_path, dataset_root):
    full_dataset = PPISDataset(esm_path=esm_path, pdb_path=pdb_path, dataset_root=dataset_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=2025)
    fold_loaders = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=4)
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders

def prepare_model_and_optimizer(config):
    model = KD_EGNN_edge(
        infeature_size=config.node_dim,
        outfeature_size=config.out_dim,
        nhidden_eg=config.nhidden_eg,
        edge_feature_size=config.edge_feature_size,
        n_eglayer=config.n_eglayer,
        nclass=config.nclass
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CosineEmbeddingLoss()
    
    return model, criterion1, criterion2, optimizer

# Training and evaluation functions
def train_one_epoch(model, criterion1, criterion2, optimizer, train_loader, device, is_running):
    model.train()
    epoch_train_loss = 0.0
    scaler = amp.GradScaler(enabled=config.use_amp)
    batch_count = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        if not is_running:
            break  # Exit immediately if stop requested
        
        # Prepare input data
        inputs = {
            'label': batch.label.to(device),
            'coors': batch.X_ca.to(device),
            'esm_feat': batch.esm_feat.to(device),
            'edge_index': batch.edge_index.to(device),
            'edge_feat': batch.edge_feat.float().to(device)
        }
        
        if len(inputs['label']) != inputs['esm_feat'].size(0):
            continue
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with amp.autocast(enabled=config.use_amp):
            # Forward pass
            pres, embedfeats = model(
                inputs['esm_feat'],
                inputs['coors'],
                inputs['edge_feat'],
                inputs['edge_index']
            )
            
            # Calculate loss
            loss = criterion1(pres[0], inputs['label'])
            
            # Contrastive loss
            contras_len = len(embedfeats[0]) // 2
            label1 = inputs['label'][:contras_len]
            label2 = inputs['label'][contras_len:contras_len*2]
            output1 = embedfeats[0][:contras_len]
            output2 = embedfeats[0][contras_len:contras_len*2]
            contras_label = torch.where(label1 != label2, -1.0, 1.0)  # 1 for similar, -1 for dissimilar
            contras_loss = criterion2(output1, output2, contras_label)
            loss += config.contrastive_weight * contras_loss
            
            # Knowledge distillation loss
            teacher_output = pres[0].detach()
            for idx in range(1, len(pres)):
                if idx < len(pres) - 1:
                    loss += CrossEntropy(pres[idx], teacher_output) * 0.3
                    loss += criterion1(pres[idx], inputs['label']) * 0.7
                else:
                    loss += criterion1(pres[idx], inputs['label'])
        
        # Backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_train_loss += loss.item()
        batch_count += 1
        torch.cuda.empty_cache()
    
    if batch_count == 0:
        return 0.0  # Avoid division by zero
    
    epoch_loss_train_avg = epoch_train_loss / batch_count
    return epoch_loss_train_avg

def evaluate(model, criterion1, val_loader, device, is_running):
    model.eval()
    all_preds = []
    all_labels = []
    epoch_val_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if not is_running:
                break  # Exit immediately if stop requested
                
            inputs = {
                'label': batch.label.to(device),
                'coors': batch.X_ca.to(device),
                'esm_feat': batch.esm_feat.to(device),
                'edge_index': batch.edge_index.to(device),
                'edge_feat': batch.edge_feat.float().to(device)
            }
            
            if len(inputs['label']) != inputs['esm_feat'].size(0):
                continue
            
            pres, _ = model(
                inputs['esm_feat'],
                inputs['coors'],
                inputs['edge_feat'],
                inputs['edge_index']
            )
            
            y_pred = pres[0]
            loss = criterion1(y_pred, inputs['label'])
            
            y_prob = torch.softmax(y_pred, dim=1)
            y_pred_np = y_prob[:, 1].cpu().numpy()
            label_np = inputs['label'].cpu().numpy()
            
            all_preds.extend(y_pred_np)
            all_labels.extend(label_np)
            epoch_val_loss += loss.item()
            batch_count += 1
    
    if batch_count == 0:
        return 0.0, [], []  # Avoid division by zero
    
    epoch_loss_val_avg = epoch_val_loss / batch_count
    return epoch_loss_val_avg, all_labels, all_preds

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'binary_acc': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'AUC': 0,
            'mcc': 0,
            'AUPRC': 0
        }
    
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = y_true
    
    results = {
        'binary_acc': metrics.accuracy_score(binary_true, binary_pred),
        'precision': metrics.precision_score(binary_true, binary_pred),
        'recall': metrics.recall_score(binary_true, binary_pred),
        'f1': metrics.f1_score(binary_true, binary_pred),
        'AUC': metrics.roc_auc_score(binary_true, y_pred),
        'mcc': metrics.matthews_corrcoef(binary_true, binary_pred)
    }
    
    # AUPRC calculation
    precisions, recalls, _ = metrics.precision_recall_curve(binary_true, y_pred)
    results['AUPRC'] = metrics.auc(recalls, precisions)
    
    return results

# Training thread
class ModelTrainer(QThread):
    update_progress = pyqtSignal(int, str)
    training_finished = pyqtSignal(str, dict)
    metrics_updated = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    
    def __init__(self, config, pdb_path, esm_path, pkl_path, output_dir):
        super().__init__()
        self.config = config
        self.pdb_path = pdb_path
        self.esm_path = esm_path
        self.pkl_path = pkl_path
        self.output_dir = output_dir
        self.is_running = True
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'auc': [],
            'auprc': []
        }
        self.device = get_device()
    
    def run(self):
        try:
            # Validate file paths
            self.update_progress.emit(0, 'Validating file paths...')
            for path in [self.pdb_path, self.esm_path, self.pkl_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
            
            # Prepare cross-validation data
            self.update_progress.emit(5, "Preparing cross-validation data...")
            fold_loaders = prepare_cross_validation(
                self.config, self.esm_path, self.pdb_path, self.pkl_path)
            
            # Prepare for training
            fold_results = []
            for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
                if not self.is_running:
                    break  # Exit immediately if stop requested
                    
                self.update_progress.emit(10 + fold_idx * 18, f'Starting training for fold {fold_idx+1}/5')
                model, criterion1, criterion2, optimizer = prepare_model_and_optimizer(self.config)
                model = model.to(self.device)
                
                fold_info = {
                    "fold": fold_idx,
                    "model_path": "",
                    "auc": 0,
                    "aupr": 0
                }
                
                best_val_auc = 0
                best_val_aupr = 0
                no_improve_epoch = 0
                
                for epoch in range(self.config.epochs):
                    if not self.is_running:
                        break  # Exit immediately if stop requested
                        
                    # Training
                    epoch_percent = 10 + fold_idx * 18 + int(epoch / self.config.epochs * 15)
                    self.update_progress.emit(
                        epoch_percent, 
                        f"Fold {fold_idx+1} | Epoch {epoch+1}/{self.config.epochs}"
                    )
                    
                    train_loss = train_one_epoch(
                        model, criterion1, criterion2, 
                        optimizer, train_loader, self.device, self.is_running
                    )
                    
                    if not self.is_running:
                        break  # Exit immediately if stop requested
                    
                    # Evaluation
                    val_loss, valid_true, valid_pred = evaluate(
                        model, criterion1, val_loader, self.device, self.is_running
                    )
                    
                    if not self.is_running:
                        break  # Exit immediately if stop requested
                    
                    results = calculate_metrics(valid_true, valid_pred)
                    
                    # Record metrics for visualization
                    self.metrics_history['train_loss'].append(train_loss)
                    self.metrics_history['val_loss'].append(val_loss)
                    self.metrics_history['auc'].append(results['AUC'])
                    self.metrics_history['auprc'].append(results['AUPRC'])
                    self.metrics_updated.emit(epoch, train_loss, val_loss)
                    
                    # Log message
                    log_msg = (
                        f"Fold {fold_idx+1} Epoch {epoch+1} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val AUC: {results['AUC']:.4f}, "
                        f"Val AUPRC: {results['AUPRC']:.4f}"
                    )
                    self.update_progress.emit(epoch_percent, log_msg)
                    
                    # Check if best model
                    if results['AUC'] > best_val_auc:
                        best_val_aupr = results['AUPRC']
                        best_val_auc = results['AUC']
                        no_improve_epoch = 0
                        
                        # Save model
                        save_path = os.path.join(self.output_dir, f"fold{fold_idx}_best_model_esm_egnn.pkl")
                        torch.save(model.state_dict(), save_path)
                        fold_info["model_path"] = save_path
                        fold_info["auc"] = best_val_auc
                        fold_info["aupr"] = best_val_aupr
                            
                        self.update_progress.emit(
                            epoch_percent, 
                            f"Saved best model! AUPRC: {best_val_aupr:.4f}"
                        )
                    else:
                        no_improve_epoch += 1
                            
                    # Early stopping check
                    if no_improve_epoch >= self.config.patience:
                        self.update_progress.emit(
                            epoch_percent, 
                            f"Early stopping triggered at epoch {epoch+1}"
                        )
                        break
                    
                    # GPU memory management
                    if self.device.type == 'cuda':
                        mem_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                        if mem_usage > 0.9 * torch.cuda.max_memory_allocated():
                            self.update_progress.emit(
                                epoch_percent, 
                                "GPU memory over 90%, clearing cache..."
                            )
                            torch.cuda.empty_cache()
                
                if not self.is_running:
                    break  # Exit immediately if stop requested
                
                fold_results.append(fold_info)
                self.update_progress.emit(
                    25 + fold_idx * 18, 
                    f"Fold {fold_idx+1} completed | Best AUPRC: {fold_info['aupr']:.4f}"
                )
            
            if self.is_running:
                # Save summary results
                results_path = os.path.join(self.output_dir, "training_summary.json")
                with open(results_path, 'w') as f:
                    json.dump(fold_results, f, indent=4)
                
                # Export models
                for fold_info in fold_results:
                    self.export_model(model, fold_info["model_path"])
                
                self.update_progress.emit(100, f"Training completed! Results saved at: {results_path}")
                self.training_finished.emit(results_path, self.metrics_history)
            else:
                self.update_progress.emit(0, "Training stopped by user")
        
        except Exception as e:
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            self.update_progress.emit(0, error_msg)
            QMessageBox.critical(None, "Training Error", error_msg)
    
    def export_model(self, model, path):
        """Export model to multiple formats"""
        try:
            # ONNX format
            dummy_input = torch.randn(1, config.node_dim).to(self.device)
            onnx_path = path.replace('.pkl', '.onnx')
            torch.onnx.export(model, dummy_input, onnx_path)
            
            # TorchScript format
            script_path = path.replace('.pkl', '.pt')
            scripted_model = torch.jit.script(model)
            scripted_model.save(script_path)
            
            self.update_progress.emit(95, f"Model exported: ONNX ({onnx_path}), TorchScript ({script_path})")
        except Exception as e:
            self.update_progress.emit(0, f"Model export failed: {str(e)}")
    
    def stop(self):
        self.is_running = False
        self.update_progress.emit(0, "Training termination request sent...")
        # Additional cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Visualization component
class TrainingVisualization(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title('Training Metrics')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        
    def update_plot(self, metrics):
        """Update loss curves"""
        self.ax.clear()
        
        if metrics['train_loss']:
            epochs = range(1, len(metrics['train_loss']) + 1)
            self.ax.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss')
            self.ax.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
            
            if metrics['auc']:
                ax2 = self.ax.twinx()
                ax2.plot(epochs, metrics['auc'], 'g--', label='AUC')
                ax2.set_ylabel('AUC')
                ax2.legend(loc='upper right')
            
            self.ax.legend(loc='upper left')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_title('Training Metrics')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            
            # Auto-adjust Y-axis range
            max_val = max(max(metrics['train_loss']), max(metrics['val_loss'])) * 1.1
            self.ax.set_ylim(0, max_val)
            
            self.draw()

# File input component
class InputGroup(QGroupBox):
    def __init__(self, title, file_types='', is_file=True):
        super().__init__(title)
        self.file_path = ""
        self.file_types = file_types
        self.is_file = is_file
        
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("Click Browse to select file or directory...")
        layout.addWidget(self.path_display, 5)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setToolTip(f"Select {title}")
        browse_btn.clicked.connect(self.select_path)
        layout.addWidget(browse_btn, 1)
    
    def select_path(self):
        if self.is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.title()} file", "", self.file_types
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, f"Select {self.title()} directory"
            )
        if path:
            self.file_path = path
            self.path_display.setText(path)
    
    def get_path(self):
        return self.file_path

# Training application interface
class ProteinTrainingApp(QWidget):
    close_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Binding Site Prediction Training Platform")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply styles
        self.apply_styles()
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Top panel - input area
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Protein Binding Site Prediction Training Platform")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #61AFEF;")
        title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(title_label)
        
        # File input group
        inputs_group = QGroupBox("Training Data Input")
        inputs_layout = QVBoxLayout()
        inputs_layout.setSpacing(15)
        inputs_layout.setContentsMargins(15, 20, 15, 15)
        
        # Input fields
        self.pdb_input = InputGroup("PDB Directory", is_file=False)
        self.esm_input = InputGroup("ESM Features Directory", is_file=False)
        self.pkl_input = InputGroup("Training Samples PKL File", "Pickle files (*.pkl);; All files (*.*)")
        self.output_input = InputGroup('Model Save Directory', is_file=False)
        
        inputs_layout.addWidget(self.pdb_input)
        inputs_layout.addWidget(self.esm_input)
        inputs_layout.addWidget(self.pkl_input)
        inputs_layout.addWidget(self.output_input)
        
        # Training parameters input
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        
        self.epoch_input = QLineEdit("50")
        self.lr_input = QLineEdit("0.0001")
        self.batch_input = QLineEdit("4")
        self.patience_input = QLineEdit("10")
        
        # Set input placeholders
        self.epoch_input.setPlaceholderText("50-200")
        self.lr_input.setPlaceholderText("0.0001-0.001")
        self.batch_input.setPlaceholderText("1-32")
        self.patience_input.setPlaceholderText("5-20")
        
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        params_layout.addWidget(self.epoch_input, 0, 1)
        params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        params_layout.addWidget(self.lr_input, 1, 1)
        params_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        params_layout.addWidget(self.batch_input, 0, 3)
        params_layout.addWidget(QLabel("Early Stopping Patience:"), 1, 2)
        params_layout.addWidget(self.patience_input, 1, 3)
        
        # Advanced parameters
        self.use_amp_cb = QCheckBox("Use Mixed Precision Training")
        self.use_amp_cb.setChecked(True)
        params_layout.addWidget(self.use_amp_cb, 2, 0, 1, 4)
        
        params_group.setLayout(params_layout)
        inputs_layout.addWidget(params_group)
        
        inputs_group.setLayout(inputs_layout)
        top_layout.addWidget(inputs_group)
        
        # Control button area
        control_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0095FF;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
        """)
        self.train_btn.setToolTip("Start model training")
        self.train_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #D84315;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #FF5722;
            }
            QPushButton:pressed {
                background-color: #BF360C;
            }
        """)
        self.stop_btn.setToolTip("Interrupt current training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        self.back_btn = QPushButton("Back to Main")
        self.back_btn.setMinimumHeight(40)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A4A4F;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #5A5A5F;
            }
            QPushButton:pressed {
                background-color: #3A3A3F;
            }
        """)
        self.back_btn.setToolTip("Return to main menu")
        self.back_btn.clicked.connect(self.return_to_main)
        
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.back_btn)
        top_layout.addLayout(control_layout)
        
        # Bottom panel - logs and visualization
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setSpacing(10)
        
        # Create bottom horizontal splitter
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Visualization panel
        vis_panel = QWidget()
        vis_layout = QVBoxLayout(vis_panel)
        vis_layout.addWidget(QLabel("Training Metrics Visualization"))
        self.visualization = TrainingVisualization()
        vis_layout.addWidget(self.visualization)
        
        # Log panel
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Training Progress:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat('Ready')
        progress_layout.addWidget(self.progress_bar, 1)
        log_layout.addLayout(progress_layout)
        
        # Log area
        log_group = QGroupBox("Training Log")
        log_group_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(300)
        self.log_display.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        log_group_layout.addWidget(self.log_display)
        log_group.setLayout(log_group_layout)
        log_layout.addWidget(log_group)
        
        # Add panels to splitter
        bottom_splitter.addWidget(vis_panel)
        bottom_splitter.addWidget(log_panel)
        bottom_splitter.setSizes([500, 500])
        bottom_layout.addWidget(bottom_splitter)
        
        # Add panels to main splitter
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([300, 500])
        
        main_layout.addWidget(splitter)
        self.trainer = None
        self.metrics_history = {}
        
        # Try to set icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "protein_icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except:
            pass
    
    def apply_styles(self):
        """Apply stylesheet"""
        self.setStyleSheet("""
            QWidget {
                background-color: #2D2D30;
            }
            QGroupBox {
                color: #E0E0E0;
                font-size: 12pt;
                font-weight: bold;
                border: 1px solid #6C6C70;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #252526;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #2D2D30;
            }
            QPushButton {
                background-color: #4A4A4F;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 11pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5A5A5F;
            }
            QPushButton:pressed {
                background-color: #3A3A3F;
            }
            QPushButton:disabled {
                background-color: #3A3A3F;
                color: #888888;
            }
            QTextEdit, QLineEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 5px;
                font-size: 10pt;
            }
            QProgressBar {
                border: 2px solid #3A3A3F;
                border-radius: 8px;
                background: #1E1E1E;
                text-align: center;
                height: 25px;
                font-size: 10pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078D7, stop:1 #00B4FF
                );
                border-radius: 6px;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 11pt;
            }
            QCheckBox {
                color: #E0E0E0;
                font-size: 10pt;
            }
        """)
    
    def update_config(self):
        """Update configuration from UI"""
        try:
            config.epochs = int(self.epoch_input.text())
            config.lr = float(self.lr_input.text())
            config.batch_size = int(self.batch_input.text())
            config.patience = int(self.patience_input.text())
            config.use_amp = self.use_amp_cb.isChecked()
        except ValueError:
            pass
    
    def save_config(self):
        """Save configuration to file"""
        self.update_config()
        try:
            config_data = {
                "epochs": config.epochs,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "patience": config.patience,
                "use_amp": config.use_amp
            }
            with open("training_config.json", "w") as f:
                json.dump(config_data, f, indent=4)
            
            self.log_display.append("Training configuration saved to training_config.json")
        except Exception as e:
            self.log_display.append(f"Failed to save configuration: {str(e)}")
    
    def start_training(self):
        """Start training process"""
        self.log_display.clear()
        
        # Get input paths
        pdb_path = self.pdb_input.get_path()
        esm_path = self.esm_input.get_path()
        pkl_path = self.pkl_input.get_path()
        output_dir = self.output_input.get_path()
        
        # Validate inputs
        errors = []
        if not pdb_path: errors.append("PDB directory")
        if not esm_path: errors.append("ESM features directory")
        if not pkl_path: errors.append("Training samples file")
        if not output_dir: errors.append("Model save directory")
        
        if errors:
            error_msg = f"Please select: {', '.join(errors)}"
            self.log_display.append(error_msg)
            self.progress_bar.setFormat("Incomplete inputs")
            QMessageBox.warning(self, "Input Error", error_msg)
            return
        
        # Update configuration
        self.update_config()
        
        # Reset UI
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing training...")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.back_btn.setEnabled(False)
        
        # Display input information
        self.log_display.append(f"Starting model training...")
        self.log_display.append(f" - PDB directory: {pdb_path}")
        self.log_display.append(f" - ESM features directory: {esm_path}")
        self.log_display.append(f" - Training samples file: {pkl_path}")
        self.log_display.append(f" - Model save directory: {output_dir}")
        self.log_display.append(f" - Training epochs: {config.epochs}")
        self.log_display.append(f" - Learning rate: {config.lr}")
        self.log_display.append(f" - Batch size: {config.batch_size}")
        self.log_display.append("-" * 80)
        
        # Create and start training thread
        self.trainer = ModelTrainer(config, pdb_path, esm_path, pkl_path, output_dir)
        self.trainer.update_progress.connect(self.update_progress)
        self.trainer.training_finished.connect(self.training_finished)
        self.trainer.metrics_updated.connect(self.update_visualization)
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'auc': [],
            'auprc': []
        }
        self.visualization.update_plot(self.metrics_history)
        
        self.trainer.start()
    
    def stop_training(self):
        """Stop training process"""
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop()
            self.log_display.append("Training stop requested...")
            self.progress_bar.setFormat("Stopping training...")
            self.stop_btn.setEnabled(False)  # Disable stop button after click
            self.back_btn.setEnabled(True)  # Enable back button
    
    def return_to_main(self):
        """Return to main menu"""
        # Stop training if running
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop()
        
        # Emit signal to return to main menu
        self.close_signal.emit()
    
    def update_progress(self, progress, message):
        """Update progress bar and log"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{progress}% - {message[:30]}{'...' if len(message) > 30 else ''}")
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())
    
    def update_visualization(self, epoch, train_loss, val_loss):
        """Update visualization chart"""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.visualization.update_plot(self.metrics_history)
    
    def training_finished(self, save_path, metrics):
        """Training completion handler"""
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Training completed!")
        self.log_display.append("-" * 80)
        self.log_display.append(f"Model successfully saved to: {save_path}")
        self.log_display.append("Training completed!")
        
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.back_btn.setEnabled(True)  # Enable back button
        
        # Update chart with complete data
        self.metrics_history = metrics
        self.visualization.update_plot(self.metrics_history)
        
        # Display results summary
        best_auc = max(metrics['auc']) if metrics['auc'] else 0
        best_auprc = max(metrics['auprc']) if metrics['auprc'] else 0
        self.log_display.append(f"Best AUC: {best_auc:.4f}, Best AUPRC: {best_auprc:.4f}")


# Run application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(74, 74, 79))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = ProteinTrainingApp()
    window.show()
    sys.exit(app.exec_())