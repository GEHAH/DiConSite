import sys
import time
import os
import random
import traceback
import warnings,json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn import metrics
sys.path.append('./task/')
from Dataset.dataset import PPISDataset
from torch_geometric.data import DataLoader
from model_block.PPIsmodels import KD_EGNN_edge
from utils import ContrastiveLoss,Seed,get_device
from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar,
    QGroupBox, QFormLayout, QLineEdit, QSplitter,QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

warnings.filterwarnings("ignore")

def prepare_cross_validation(config,esm_path,pdb_path,dataset_root):
    # 加载完整数据集
    full_dataset = PPISDataset(esm_path=esm_path,
                            pdb_path=pdb_path,dataset_root=dataset_root)
    # 初始化五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2025)
    # 存储所有fold的DataLoader
    fold_loaders = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Preparing Fold {fold_idx + 1}/5")
        
        # 创建子集数据集
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # 创建DataLoader
        train_loader = DataLoader(train_subset, batch_size=config.batch_size,shuffle=True,num_workers=10)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size,shuffle=False,num_workers=10)
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders

def prepare_model_and_optimizer(config):
    model = KD_EGNN_edge(infeature_size=config.node_dim,outfeature_size=config.out_dim,
                     nhidden_eg=config.nhidden_eg,edge_feature_size=config.edge_feature_size,
                    n_eglayer=config.n_eglayer,nclass=config.nclass)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = ContrastiveLoss()
    return model,criterion1,criterion2,optimizer

def train_one_epoch(model,criterion1,criterion2,optimizer,train_loader):
    device = get_device()
    model = model.to(device)
    model.train()
    epoch_train_loss = 0.
    n = 0
    for batch in tqdm(train_loader,desc='Training'):
        inputs = {
                'label': batch.label.to(device),
                'coors': batch.X_ca.to(device),
                'esm_feat': batch.esm_feat.to(device),
                'edge_index': batch.edge_index.to(device),
                'edge_feat': batch.edge_feat.float().to(device)
            }
        if len(inputs['label']) != inputs['esm_feat'].size(0):
            continue
        #重置梯度
        optimizer.zero_grad()

        #前向传播
        pres, embedfeats = model(
                inputs['esm_feat'],
                inputs['coors'],
                inputs['edge_feat'],
                inputs['edge_index']
            )
        z = embedfeats[0]  # 教师模型的嵌入特征
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        layer_list = []
        teacher_feat_size = embedfeats[0].size(1)
        for index in range(1, len(pres)):
            student_feature_size = embedfeats[index].size(1)
            layer_list.append(nn.Linear(student_feature_size, teacher_feat_size))
        model.adaptation_layers = nn.ModuleList(layer_list)
        model.adaptation_layers.to(device)
        loss = torch.FloatTensor([0.]).to(device)
        loss += criterion1(pres[0], inputs['label'])
        contras_len = len(z)//2
        label1 = inputs['label'][:contras_len]
        label2 = inputs['label'][contras_len:contras_len*2]
        output1 = z[:contras_len]
        output2 = z[contras_len:contras_len*2]
        contras_label = (label1 != label2).float()
        contras_loss = criterion2(output1,output2,contras_label)
        loss += contras_loss
        teacher_output = pres[0].detach()
        teacher_feature = embedfeats[0].detach()

        for index in range(1, len(pres)-1):
            loss += CrossEntropy(pres[index], teacher_output) * 0.3 # KL_loss soft loss
            loss += criterion1(pres[index],inputs['label']) * (1 - 0.3)  # hard loss 学生自己的
            if index != 1:
                loss += torch.dist(model.adaptation_layers[index - 1](embedfeats[index]), teacher_feature) * 0.03
        
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        n += 1
        torch.cuda.empty_cache()
    epoch_loss_train_avg = epoch_train_loss / n
    return epoch_loss_train_avg
def evaluate(model,criterion1,val_loader):
    device = get_device()
    model = model.to(device)
    model.eval()
    n = 0
    all_preds = []
    all_labels = []
    epoch_val_loss = 0.
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = {
                'label': batch.label.to(device),
                'coors': batch.X_ca.to(device),
                'esm_feat': batch.esm_feat.to(device),
                'edge_index': batch.edge_index.to(device),
                'edge_feat': batch.edge_feat.float().to(device)
            }
            if len(inputs['label']) != inputs['esm_feat'].size(0):
                continue
            pres, embedfeats = model(
                    inputs['esm_feat'],
                    inputs['coors'],
                    inputs['edge_feat'],
                    inputs['edge_index']
                )
            y_pred = pres[0]
            loss = criterion1(y_pred,inputs['label'])
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.cpu().detach().numpy()
            label = inputs['label'].cpu().detach().numpy()
            all_preds += [pred[1] for pred in y_pred]
            all_labels += list(label)
            epoch_val_loss += loss.item()
            n += 1
    epoch_loss_val_avg = epoch_val_loss / n
    
    return epoch_loss_val_avg, all_labels,all_preds

def analysis(y_true,y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = y_true
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': thresholds
    }
    return results

def train(model,criterion1,criterion2,optimizer,train_loader, val_loader,Fold,config):
    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0
    model = model.cuda()
    no_improve_epoch = 0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        epoch_loss_train_avg = train_one_epoch(model,criterion1,criterion2,optimizer,train_loader)
        print(f"Train Loss: {epoch_loss_train_avg}")
        print('Evaluating val set...')
        epoch_loss_val_avg, valid_true, valid_pred = evaluate(model,criterion1,val_loader)
        results = analysis(valid_true, valid_pred)
        print(f"Val Loss: {epoch_loss_val_avg}")
        print(f'Val binary acc: {results["binary_acc"]}')
        print(f'Val precision: {results["precision"]}')
        print(f'Val recall: {results["recall"]}')
        print(f'Val f1: {results["f1"]}')
        print(f"Val AUC: {results['AUC']}")
        print(f"Val AUPRC: {results['AUPRC']}")
        print(f"Val mcc: {results['mcc']}")
        if best_val_auc < results['AUC']:
            best_val_aupr = results['AUPRC']
            best_epoch = epoch
            no_improve_epoch = 0
            best_val_auc = results['AUC']
            best_val_aupr = results['AUPRC']
            torch.save(model.state_dict(), os.path.join('/owenbhe/buddy1/lqszchen/PPIs/PPIS_sekd/results_new/DNA/' +str(Fold)+ '_best_model_esm_egnn.pkl'))
            print(f"New best model saved! AUPRC improved to {best_val_aupr:.4f}")
        else:
            no_improve_epoch += 1
            print(f"AUPRC did not improve for {no_improve_epoch}/{config.patience} epochs")
        if no_improve_epoch >= config.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best_epoch: {best_epoch}. Best AUPRC: {best_val_aupr:.4f}")
            break

#加载参数
class Args:
    def __init__(self) -> None:
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
config = Args()
def CrossEntropy(outputs, targets,temperature=3):
	log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
	softmax_targets = F.softmax(targets / temperature, dim=1)
	return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
# 模拟的模型训练类
class ModelTrainer(QThread):
    update_progress = pyqtSignal(int, str)  # 进度百分比, 日志消息
    training_finished = pyqtSignal(str)     # 保存路径
    # training_failed = pyqtSignal(str)

    def __init__(self, pdb_path, esm_path, pkl_path,model_path,config):
        super().__init__()
        self.pdb_path = pdb_path
        self.esm_path = esm_path
        self.pkl_path = pkl_path
        self.model_path = model_path
        self.config = config
        self.is_running = True

    def run(self):
        try:
            # 检查文件路径
            self.update_progress.emit(0,'Verify the file path...')
            for path in [self.pdb_path, self.esm_path, self.pkl_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing files: {path}")
            #加载交叉验证数据
            self.update_progress.emit(5, "Prepare the cross-validation data...")
            fold_loaders = prepare_cross_validation(self.config,self.esm_path,self.pdb_path,self.pkl_path)

            #准备训练
            self.update_progress.emit(10, "Initialize the model and optimizer...")
            fold_results = []
            for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
                if not self.is_running:
                    break
                self.update_progress.emit(10+fold_idx*18,f'Start training Fold {fold_idx+1}/5')
                model, criterion1, criterion2, optimizer = prepare_model_and_optimizer(self.config)
                fold_info = {
                        "fold": fold_idx,
                        "model_path": "",
                        "auc": 0,
                        "aupr": 0
                    }
                best_epoch = 0
                best_val_auc = 0
                best_val_aupr = 0
                no_improve_epoch = 0 
                for epoch in range(self.config.epochs):
                    if not self.is_running:
                        break

                    epoch_percent = 10 + fold_idx * 18 + int(epoch / self.config.epochs * 15)
                    msg = f"Fold {fold_idx+1} | Epoch {epoch + 1}/{self.config.epochs}"
                    self.update_progress.emit(epoch_percent, msg) 
                    epoch_loss_train_avg = train_one_epoch(model, criterion1, criterion2, 
                                                            optimizer, train_loader)
                    # 验证
                    epoch_loss_val_avg, valid_true, valid_pred = evaluate(model, criterion1, val_loader)
                    results = analysis(valid_true, valid_pred)
                    # 记录日志
                    log_msg = (f"Fold {fold_idx+1} Epoch {epoch+1} - "
                                f"Train Loss: {epoch_loss_train_avg:.4f}, "
                                f"Val Loss: {epoch_loss_val_avg:.4f}, "
                                f"Val AUC: {results['AUC']:.4f}, "
                                f"Val AUPRC: {results['AUPRC']:.4f}")
                    self.update_progress.emit(epoch_percent, log_msg)
                    if results['AUC'] > best_val_auc:
                        best_val_aupr = results['AUPRC']
                        best_epoch = epoch
                        no_improve_epoch = 0
                        best_val_auc = results['AUC']
                        best_val_aupr = results['AUPRC']
                            
                        # 保存模型
                        save_path = os.path.join(self.model_path, f"fold{fold_idx}_best_model_esm_egnn.pkl")
                        torch.save(model.state_dict(), save_path)
                        fold_info["model_path"] = save_path
                        fold_info["auc"] = best_val_auc
                        fold_info["aupr"] = best_val_aupr
                            
                        self.update_progress.emit(epoch_percent, 
                                                    f"New best model saved! AUPRC: {best_val_aupr:.4f}")
                    else:
                        no_improve_epoch += 1
                            
                    # 早停检查
                    if no_improve_epoch >= self.config.patience:
                        self.update_progress.emit(epoch_percent, 
                                                    f"Early stopping at epoch {epoch+1}")
                        break
                fold_results.append(fold_info)
                self.update_progress.emit(25 + fold_idx * 18, 
                                            f"Fold {fold_idx+1} Finish | Best AUPRC: {fold_info['aupr']:.4f}")
                
            if self.is_running:
                if self.is_running:
                # 保存汇总结果
                    results_path = "training_summary.json"
                    with open(results_path, 'w') as f:
                        json.dump(fold_results, f, indent=4)
                        
                    self.update_progress.emit(100, f"Training is complete! Results are saved in: {results_path}")
                    self.training_finished.emit(results_path)
                
        except Exception as e:
            self.update_progress.emit(0, f"error: {str(e)}")
            # self.training_failed.emit(f"error: {str(e)}")
            # traceback.print_exc()

    def stop(self):
        self.is_running = False
        self.update_progress.emit(0, "Training termination request sent...")

# 美观的输入字段组
class InputGroup(QGroupBox):
    def __init__(self, title, file_types='',is_file=True):
        super().__init__(title)
        self.file_path = ""
        self.file_types = file_types
        self.is_file = is_file
        
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # 文件路径显示
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("Click the Browse button to select the file..")
        layout.addWidget(self.path_display, 5)
        
        # 浏览按钮
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.select_file)
        layout.addWidget(browse_btn, 1)
    
    def select_file(self):
        if self.is_file:
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.title()} file", "", self.file_types
            )
        else:
            file_path = QFileDialog.getExistingDirectory(
                self, f"Select {self.title()} Director"
            )
        if file_path:
            self.file_path = file_path
            self.path_display.setText(file_path)
    
    def get_path(self):
        return self.file_path

# 主界面类
class ProteinTrainingApp(QMainWindow):
    close_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein binding site prediction training platform")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QGroupBox {
                color: #E0E0E0;
                font-size: 12pt;
                font-weight: bold;
                border: 1px solid #6C6C70;
                border-radius: 5px;
                margin-top: 1ex;
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
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #404040;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 5px;
                text-align: center;
                background-color: #1E1E1E;
            }
            QProgressBar::chunk {
                background-color: #0078D7;
                width: 10px;
            }
            QLineEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 11pt;
            }
        """)
        
        # 初始化训练线程
        self.trainer = None
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 创建分隔器
        splitter = QSplitter(Qt.Vertical)
        
        # 顶部面板 - 输入区域
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("Protein binding site prediction training platform")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #61AFEF;")
        title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(title_label)
        
        # 文件输入组
        inputs_group = QGroupBox("Training data input")
        inputs_layout = QVBoxLayout()
        inputs_layout.setSpacing(15)
        inputs_layout.setContentsMargins(15, 20, 15, 15)
        
        # 创建四个输入字段
        self.pdb_input = InputGroup("PDB file", is_file=False)
        self.esm_input = InputGroup("ESM Feature File", is_file=False)
        self.pkl_input = InputGroup("Training sample PKL file", "Pickle file (*.pkl);; All files (*.*)")
        self.output_input = InputGroup('Model save directory',is_file=False)
        
        inputs_layout.addWidget(self.pdb_input)
        inputs_layout.addWidget(self.esm_input)
        inputs_layout.addWidget(self.pkl_input)
        inputs_layout.addWidget(self.output_input)
        
        # 训练参数输入
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Number of training rounds:"))
        
        self.epoch_input = QLineEdit("60")
        self.epoch_input.setMaximumWidth(80)
        params_layout.addWidget(self.epoch_input)
        params_layout.addStretch()
        
        inputs_layout.addLayout(params_layout)
        inputs_group.setLayout(inputs_layout)
        
        top_layout.addWidget(inputs_group)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Start training")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.train_btn.setStyleSheet("background-color: #0078D7;")
        self.train_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton("Stop training")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("background-color: #D84315;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.stop_btn)
        top_layout.addLayout(control_layout)
        
        # 底部面板 - 日志和进度
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setSpacing(10)
        
        # 进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Training Progress:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat('Be all set')
        progress_layout.addWidget(self.progress_bar, 1)
        
        bottom_layout.addLayout(progress_layout)
        
        # 日志区域
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(200)
        
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        
        bottom_layout.addWidget(log_group, 1)
        
        # 添加面板到分隔器
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([400, 300])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
    
    def start_training(self):
        # 获取输入路径
        pdb_path = self.pdb_input.get_path()
        esm_path = self.esm_input.get_path()
        pkl_path = self.pkl_input.get_path()
        output_dir = self.output_input.get_path()
        
        # 验证输入
        errors = []
        if not pdb_path: errors.append("PDB file")
        if not esm_path: errors.append("ESM feature file")
        if not pkl_path: errors.append("Training sample PKL file")
        
        if errors:
            self.log_display.append(f"Correct translation: Please select first{', '.join(errors)}!")
            self.progress_bar.setFormat("Input is incomplete.")
            return
        
        # 获取训练轮数
        try:
            epochs = int(self.epoch_input.text())
            if epochs <= 0:
                raise ValueError("The number of rounds must be a positive integer.")
        except ValueError as e:
            self.log_display.append(f"errors: {str(e)}")
            self.progress_bar.setFormat("parameter error")
            return
        
        # 重置界面
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("初始化训练...")
        self.log_display.clear()
        
        # 显示输入信息
        self.log_display.append(f"Starting to train the model...")
        self.log_display.append(f" - PDB file: {pdb_path}")
        self.log_display.append(f" - ESM feature file: {esm_path}")
        self.log_display.append(f" - Training sample file: {pkl_path}")
        self.log_display.append(f" - Model saving directory: {output_dir}")
        self.log_display.append(f" - Number of training rounds: {epochs}")
        self.log_display.append("-" * 80)
        
        # 创建并启动训练线程
        self.trainer = ModelTrainer(pdb_path, esm_path, pkl_path, output_dir, config)
        self.trainer.update_progress.connect(self.update_progress)
        self.trainer.training_finished.connect(self.training_finished)
        # self.trainer.training_failed.connect(self.training_failed)
        
        self.trainer.start()
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def stop_training(self):
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop()
            self.log_display.append("Training has been suspended.")
            self.progress_bar.setFormat("Training interruption")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{progress}% - {message[:30]}{'...' if len(message)>30 else ''}")
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum())
    
    def training_finished(self, save_path):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Training completed!")
        self.log_display.append("-" * 80)
        self.log_display.append(f"The model has been successfully saved as: {save_path}")
        self.log_display.append("Training completed!")
        
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # def training_failed(self, error_message):
    #     self.log_display(f"Training failed: {error_message}")
    #     self.progress_bar.setFormat('Training failed')
        
    #     self.train_btn.setEnabled(True)
    #     self.stop_btn.setEnabled(False)

# 运行应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
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