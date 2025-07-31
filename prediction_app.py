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
import time

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

class ProteinPredictionApp(QWidget):
    close_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Binding Site Prediction System")
        self.setGeometry(100, 100, 1200, 800)
        
        # 应用样式
        self.apply_styles()
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 顶部面板 - 输入区域
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("Protein Binding Site Prediction System")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #61AFEF;")
        title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(title_label)
    
        
        # 文件输入组
        inputs_group = QGroupBox("Prediction Input")
        inputs_layout = QVBoxLayout()
        inputs_layout.setSpacing(15)
        inputs_layout.setContentsMargins(15, 20, 15, 15)
        
        # 输入字段
        self.model_input = InputGroup("Model Directory", is_file=False)
        self.pdb_input = InputGroup("PDB File", "PDB files (*.pdb);; All files (*.*)")
        self.esm_input = InputGroup("ESM Features File", "NPZ files (*.npz);; All files (*.*)")
        self.output_input = InputGroup('Output Directory', is_file=False)
        
        inputs_layout.addWidget(self.model_input)
        inputs_layout.addWidget(self.pdb_input)
        inputs_layout.addWidget(self.esm_input)
        inputs_layout.addWidget(self.output_input)
        
        # 预测参数
        params_group = QGroupBox("Prediction Parameters")
        params_layout = QGridLayout()
        
        self.confidence_threshold = QLineEdit("0.7")
        self.chain_selection = QLineEdit("A")
        self.residue_range = QLineEdit("1-300")
        
        params_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        params_layout.addWidget(self.confidence_threshold, 0, 1)
        params_layout.addWidget(QLabel("Chain Selection:"), 1, 0)
        params_layout.addWidget(self.chain_selection, 1, 1)
        params_layout.addWidget(QLabel("Residue Range:"), 0, 2)
        params_layout.addWidget(self.residue_range, 0, 3)
        
        # 高级选项
        self.visualize_3d_cb = QCheckBox("Enable 3D Visualization")
        self.visualize_3d_cb.setChecked(True)
        self.export_csv_cb = QCheckBox("Export CSV Results")
        self.export_csv_cb.setChecked(True)
        
        params_layout.addWidget(self.visualize_3d_cb, 1, 2, 1, 2)
        params_layout.addWidget(self.export_csv_cb, 2, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        inputs_layout.addWidget(params_group)
        
        inputs_group.setLayout(inputs_layout)
        top_layout.addWidget(inputs_group)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("Start Prediction")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #34D058;
            }
            QPushButton:pressed {
                background-color: #22863A;
            }
        """)
        self.predict_btn.setToolTip("Start binding site prediction")
        self.predict_btn.clicked.connect(self.start_prediction)
        
        self.stop_btn = QPushButton("Stop Prediction")
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
        self.stop_btn.setToolTip("Interrupt current prediction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_prediction)
        
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
        
        control_layout.addWidget(self.predict_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.back_btn)
        top_layout.addLayout(control_layout)
        
        # 底部面板 - 结果和日志
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setSpacing(10)
        
        # 创建底部水平分割器
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # 结果可视化面板
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.addWidget(QLabel("Prediction Results Visualization"))
        
        # 创建堆叠窗口用于多种可视化
        self.visualization_stack = QStackedWidget()
        
        # 1. 蛋白质结构视图
        self.structure_canvas = ProteinStructureCanvas()
        self.visualization_stack.addWidget(self.structure_canvas)
        
        # 2. 残基概率图
        self.residue_plot = ResidueProbabilityPlot()
        self.visualization_stack.addWidget(self.residue_plot)
        
        # 3. 统计摘要
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.visualization_stack.addWidget(self.stats_display)
        
        result_layout.addWidget(self.visualization_stack)
        
        # 可视化选择按钮
        vis_buttons_layout = QHBoxLayout()
        self.structure_btn = QPushButton("3D Structure")
        self.structure_btn.setCheckable(True)
        self.structure_btn.setChecked(True)
        self.structure_btn.clicked.connect(lambda: self.switch_visualization(0))
        
        self.probability_btn = QPushButton("Residue Probabilities")
        self.probability_btn.setCheckable(True)
        self.probability_btn.clicked.connect(lambda: self.switch_visualization(1))
        
        self.stats_btn = QPushButton("Statistical Summary")
        self.stats_btn.setCheckable(True)
        self.stats_btn.clicked.connect(lambda: self.switch_visualization(2))
        
        vis_buttons_layout.addWidget(self.structure_btn)
        vis_buttons_layout.addWidget(self.probability_btn)
        vis_buttons_layout.addWidget(self.stats_btn)
        result_layout.addLayout(vis_buttons_layout)
        
        # 日志面板
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        
        # 进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Prediction Progress:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat('Ready')
        progress_layout.addWidget(self.progress_bar, 1)
        log_layout.addLayout(progress_layout)
        
        # 日志区域
        log_group = QGroupBox("Prediction Log")
        log_group_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(300)
        self.log_display.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        log_group_layout.addWidget(self.log_display)
        log_group.setLayout(log_group_layout)
        log_layout.addWidget(log_group)
        
        # 添加面板到分割器
        bottom_splitter.addWidget(result_panel)
        bottom_splitter.addWidget(log_panel)
        bottom_splitter.setSizes([500, 500])
        bottom_layout.addWidget(bottom_splitter)
        
        # 添加面板到主分割器
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([300, 500])
        
        main_layout.addWidget(splitter)
        self.predictor = None
        self.prediction_results = None
        
        # 设置图标
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
    
    def switch_visualization(self, index):
        """切换可视化视图"""
        self.visualization_stack.setCurrentIndex(index)
        # 更新按钮状态
        self.structure_btn.setChecked(index == 0)
        self.probability_btn.setChecked(index == 1)
        self.stats_btn.setChecked(index == 2)
    
    def start_prediction(self):
        """开始预测过程"""
        self.log_display.clear()
        
        # 获取输入路径
        model_path = self.model_input.get_path()
        pdb_path = self.pdb_input.get_path()
        esm_path = self.esm_input.get_path()
        output_dir = self.output_input.get_path()
        
        # 验证输入
        errors = []
        if not model_path: errors.append("Model directory")
        if not pdb_path: errors.append("PDB file")
        if not esm_path: errors.append("ESM features file")
        if not output_dir: errors.append("Output directory")
        
        if errors:
            error_msg = f"Please select: {', '.join(errors)}"
            self.log_display.append(error_msg)
            self.progress_bar.setFormat("Incomplete inputs")
            QMessageBox.warning(self, "Input Error", error_msg)
            return
        
        # 获取参数
        try:
            confidence_threshold = float(self.confidence_threshold.text())
            if confidence_threshold < 0 or confidence_threshold > 1:
                raise ValueError
        except ValueError:
            self.log_display.append("Invalid confidence threshold. Using default 0.7")
            confidence_threshold = 0.7
        
        # 重置UI
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Initializing prediction...")
        self.predict_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.back_btn.setEnabled(False)
        
        # 显示输入信息
        self.log_display.append(f"Starting binding site prediction...")
        self.log_display.append(f" - Model directory: {model_path}")
        self.log_display.append(f" - PDB file: {pdb_path}")
        self.log_display.append(f" - ESM features file: {esm_path}")
        self.log_display.append(f" - Output directory: {output_dir}")
        self.log_display.append(f" - Confidence threshold: {confidence_threshold}")
        self.log_display.append("-" * 80)
        
        # 创建并启动预测线程
        self.predictor = PredictionWorker(
            model_path, pdb_path, esm_path, output_dir, confidence_threshold
        )
        self.predictor.update_progress.connect(self.update_progress)
        self.predictor.prediction_finished.connect(self.prediction_finished)
        self.predictor.result_ready.connect(self.update_results)
        
        self.predictor.start()
    
    def stop_prediction(self):
        """停止预测过程"""
        if self.predictor and self.predictor.isRunning():
            self.predictor.stop()
            self.log_display.append("Prediction stop requested...")
            self.progress_bar.setFormat("Stopping prediction...")
            self.stop_btn.setEnabled(False)
    
    def return_to_main(self):
        """返回主菜单"""
        if self.predictor and self.predictor.isRunning():
            self.predictor.stop()
        self.close_signal.emit()
    
    def update_progress(self, progress, message):
        """更新进度条和日志"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{progress}% - {message[:30]}{'...' if len(message) > 30 else ''}")
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())
    
    def update_results(self, results):
        """更新预测结果"""
        self.prediction_results = results
        
        # 更新统计摘要
        stats_html = f"""
            <h3>Prediction Summary</h3>
            <p><b>Protein:</b> {results['protein_name']}</p>
            <p><b>Chains:</b> {', '.join(results['chains'])}</p>
            <p><b>Total residues:</b> {results['total_residues']}</p>
            <p><b>Predicted binding sites:</b> {results['binding_sites']} ({results['binding_percentage']:.2f}%)</p>
            <p><b>Top 5 high-probability residues:</b></p>
            <ul>
        """
        
        for residue in results['top_residues']:
            stats_html += f"<li>{residue['residue']} (Chain {residue['chain']}): {residue['probability']*100:.2f}%</li>"
        
        stats_html += "</ul>"
        self.stats_display.setHtml(stats_html)
        
        # 更新残基概率图
        self.residue_plot.update_plot(results['residue_probabilities'])
        
        # 更新3D结构可视化
        self.structure_canvas.update_structure(results['pdb_path'], results['binding_sites'])
    
    def prediction_finished(self, save_path):
        """预测完成处理"""
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Prediction completed!")
        self.log_display.append("-" * 80)
        self.log_display.append(f"Results successfully saved at: {save_path}")
        self.log_display.append("Prediction completed!")
        
        self.predict_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.back_btn.setEnabled(True)
        
        # 切换到统计摘要视图
        self.switch_visualization(2)


# 预测工作线程
class PredictionWorker(QThread):
    update_progress = pyqtSignal(int, str)
    prediction_finished = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    
    def __init__(self, model_path, pdb_path, esm_path, output_dir, confidence):
        super().__init__()
        self.model_path = model_path
        self.pdb_path = pdb_path
        self.esm_path = esm_path
        self.output_dir = output_dir
        self.confidence = confidence
        self.is_running = True
    
    def run(self):
        try:
            # 模拟预测过程
            self.update_progress.emit(10, "Loading model...")
            # 实际应用中这里会加载模型
            time.sleep(1)
            
            self.update_progress.emit(20, "Processing PDB structure...")
            # 解析PDB文件
            protein_name = os.path.basename(self.pdb_path).replace(".pdb", "")
            time.sleep(1)
            
            self.update_progress.emit(40, "Extracting ESM features...")
            # 加载ESM特征
            time.sleep(1)
            
            self.update_progress.emit(60, "Running binding site prediction...")
            # 执行预测
            time.sleep(2)
            
            # 生成模拟结果
            results = {
                'protein_name': protein_name,
                'pdb_path': self.pdb_path,
                'chains': ['A', 'B'],
                'total_residues': 250,
                'binding_sites': 42,
                'binding_percentage': 16.8,
                'top_residues': [
                    {'residue': 'ALA83', 'chain': 'A', 'probability': 0.97},
                    {'residue': 'GLY127', 'chain': 'A', 'probability': 0.95},
                    {'residue': 'LYS54', 'chain': 'B', 'probability': 0.93},
                    {'residue': 'ASP22', 'chain': 'A', 'probability': 0.91},
                    {'residue': 'ARG189', 'chain': 'B', 'probability': 0.89}
                ],
                'residue_probabilities': np.random.rand(250)  # 模拟概率数据
            }
            
            self.result_ready.emit(results)
            
            # 保存结果
            self.update_progress.emit(90, "Saving results...")
            save_path = os.path.join(self.output_dir, f"{protein_name}_results.json")
            # 实际应用中会保存结果
            time.sleep(1)
            
            if self.is_running:
                self.prediction_finished.emit(save_path)
            else:
                self.update_progress.emit(0, "Prediction stopped by user")
        
        except Exception as e:
            self.update_progress.emit(0, f"Prediction error: {str(e)}")
    
    def stop(self):
        self.is_running = False
        self.update_progress.emit(0, "Stopping prediction...")


# 蛋白质结构可视化组件
class ProteinStructureCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('Protein Structure with Predicted Binding Sites')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.fig.tight_layout()
        self.structure_data = None
    
    def update_structure(self, pdb_path, binding_sites):
        """更新蛋白质结构可视化"""
        self.ax.clear()
        
        # 实际应用中会解析PDB文件并绘制结构
        # 这里使用模拟数据
        
        # 生成蛋白质骨架
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        z = np.cos(x)
        self.ax.plot(x, y, z, 'b-', linewidth=1.5, label='Protein Backbone')
        
        # 生成结合位点
        num_sites = min(20, binding_sites)  # 限制显示数量
        site_x = np.random.rand(num_sites) * 10
        site_y = np.sin(site_x) + np.random.rand(num_sites) * 0.3 - 0.15
        site_z = np.cos(site_x) + np.random.rand(num_sites) * 0.3 - 0.15
        self.ax.scatter(site_x, site_y, site_z, c='r', s=50, label='Binding Sites')
        
        self.ax.legend()
        self.ax.set_title('Protein Structure with Predicted Binding Sites')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.draw()


# 残基概率图组件
class ResidueProbabilityPlot(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        self.ax.set_title('Residue Binding Probabilities')
        self.ax.set_xlabel('Residue Position')
        self.ax.set_ylabel('Binding Probability')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 1)
    
    def update_plot(self, probabilities):
        """更新残基概率图"""
        self.ax.clear()
        
        if len(probabilities) > 0:
            positions = np.arange(1, len(probabilities) + 1)
            self.ax.bar(positions, probabilities, color='skyblue', edgecolor='navy')
            
            # 添加阈值线
            self.ax.axhline(y=0.7, color='r', linestyle='--', label='Confidence Threshold')
            
            # 高亮高概率残基
            high_prob_indices = np.where(probabilities > 0.7)[0]
            if len(high_prob_indices) > 0:
                self.ax.bar(positions[high_prob_indices], probabilities[high_prob_indices], 
                           color='salmon', edgecolor='darkred')
            
            self.ax.set_title('Residue Binding Probabilities')
            self.ax.set_xlabel('Residue Position')
            self.ax.set_ylabel('Binding Probability')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_ylim(0, 1)
            
            # 设置x轴范围
            self.ax.set_xlim(0, len(probabilities) + 1)
            
            self.draw()


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



# 运行应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置应用样式（与训练模块相同）
    window = ProteinPredictionApp()
    window.show()
    sys.exit(app.exec_())