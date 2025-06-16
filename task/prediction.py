import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar,
    QGroupBox, QFormLayout, QLineEdit, QSplitter, QScrollArea, 
    QSizePolicy, QTextBrowser, QGridLayout, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QTextCursor, QTextCharFormat, QBrush

# 模拟的预测类
class BindingSitePredictor(QThread):
    update_progress = pyqtSignal(int, str)  # 进度百分比, 日志消息
    prediction_complete = pyqtSignal(str, dict)     # 预测结果 (序列, 结合位点字典)

    def __init__(self, sequence, pdb_path, esm_path, model_path):
        super().__init__()
        self.sequence = sequence
        self.pdb_path = pdb_path
        self.esm_path = esm_path
        self.model_path = model_path
        self.is_running = True

    def run(self):
        try:
            # 验证输入
            if not self.sequence or not self.sequence.strip():
                raise ValueError("蛋白质序列不能为空")
                
            # 模拟验证其他文件
            self.update_progress.emit(10, "正在验证输入文件...")
            
            if not self.pdb_path:
                self.update_progress.emit(0, "警告: 未提供PDB文件，预测可能不准确")
            if not self.esm_path:
                self.update_progress.emit(0, "警告: 未提供ESM特征文件，预测可能不准确")
            if not self.model_path:
                raise FileNotFoundError("模型权重文件未提供")
            
            # 模拟加载过程
            self.update_progress.emit(25, "加载模型权重...")
            
            # 模拟处理过程
            self.update_progress.emit(40, "提取蛋白质特征...")
            
            # 模拟预测过程
            self.update_progress.emit(50, "预测结合位点...")
            
            # 生成模拟的预测结果
            sites = {}
            position_notes = {}
            
            # 随机选择一些位置作为结合位点
            seq_len = len(self.sequence)
            num_sites = max(1, int(seq_len * 0.2))  # 20%的位置作为结合位点
            
            for _ in range(num_sites):
                pos = random.randint(0, seq_len - 1)
                confidence = round(random.uniform(0.6, 0.98), 2)
                sites[pos] = confidence
                
                # 添加注释（模拟）
                annotations = ["催化活性位点", "配体结合域", "金属离子结合位点", 
                              "变构调节位点", "蛋白质结合界面", "底物特异性位点"]
                position_notes[pos] = random.choice(annotations)
            
            # 模拟计算时间
            for i in range(5):
                if not self.is_running:
                    return
                progress = 60 + i * 8
                self.update_progress.emit(progress, f"处理预测结果({i+1}/5)...")
            
            if self.is_running:
                # 返回预测结果
                result = {
                    "binding_sites": sites,
                    "position_notes": position_notes,
                    "confidence_scores": sites,
                    "sequence": self.sequence
                }
                self.prediction_complete.emit("预测完成", result)
                
        except Exception as e:
            self.update_progress.emit(0, f"错误: {str(e)}")

    def stop(self):
        self.is_running = False

# 美观的输入字段组
class InputGroup(QGroupBox):
    def __init__(self, title, file_types, is_text_area=False):
        super().__init__(title)
        self.file_path = ""
        self.file_types = file_types
        self.is_text_area = is_text_area
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if self.is_text_area:
            # 文本输入区域
            self.text_input = QTextEdit()
            self.text_input.setPlaceholderText(f"在此输入{title}...")
            self.text_input.setAcceptRichText(False)
            self.text_input.setStyleSheet("""
                QTextEdit {
                    background-color: #1E1E1E;
                    color: #D4D4D4;
                    border: 1px solid #404040;
                    border-radius: 4px;
                    font-family: Consolas, 'Courier New', monospace;
                    font-size: 11pt;
                }
            """)
            layout.addWidget(self.text_input)
            
            # 文件加载按钮
            file_btn = QPushButton("从文件加载序列")
            file_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4A4A4F;
                    color: white;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #5A5A5F;
                }
            """)
            file_btn.clicked.connect(self.load_from_file)
            layout.addWidget(file_btn)
        else:
            # 文件路径显示
            path_layout = QHBoxLayout()
            self.path_display = QLineEdit()
            self.path_display.setReadOnly(True)
            self.path_display.setPlaceholderText("点击浏览按钮选择文件...")
            path_layout.addWidget(self.path_display, 5)
            
            # 浏览按钮
            browse_btn = QPushButton("浏览...")
            browse_btn.setFixedWidth(80)
            browse_btn.clicked.connect(self.select_file)
            path_layout.addWidget(browse_btn, 1)
            layout.addLayout(path_layout)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"选择 {self.title()} 文件", "", self.file_types
        )
        if file_path:
            self.file_path = file_path
            self.path_display.setText(file_path)
    
    def load_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"加载 {self.title()}", "", "文本文件 (*.txt *.fasta *.seq);;所有文件 (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # 简化FASTA格式处理（移除标题行）
                    if content.startswith(">"):
                        content = "".join(content.split("\n")[1:])
                    self.text_input.setPlainText(content)
            except Exception as e:
                self.text_input.setPlainText(f"错误: 无法读取文件 - {str(e)}")
    
    def get_path(self):
        return self.file_path if not self.is_text_area else ""
    
    def get_text(self):
        return self.text_input.toPlainText().strip() if self.is_text_area else ""

# 主界面类
class BindingSitePredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("蛋白质结合位点预测工具")
        self.setGeometry(100, 100, 1200, 800)
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
            QTextEdit, QTextBrowser {
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
            QTableWidget {
                background-color: #1E1E1E;
                color: #D4D4D4;
                gridline-color: #404040;
                font-size: 10pt;
                border: 1px solid #404040;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #3A3A3A;
                color: #E0E0E0;
                padding: 4px;
                border: 1px solid #404040;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        
        # 初始化预测线程
        self.predictor = None
        self.prediction_result = None
        
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
        title_label = QLabel("蛋白质结合位点预测")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #61AFEF;")
        title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(title_label)
        
        # 创建网格布局用于输入
        input_grid = QGridLayout()
        input_grid.setSpacing(15)
        input_grid.setContentsMargins(10, 10, 10, 10)
        
        # 蛋白质序列输入
        sequence_group = InputGroup("蛋白质序列", "", is_text_area=True)
        sequence_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        input_grid.addWidget(sequence_group, 0, 0, 2, 1)
        
        # PDB文件输入
        pdb_group = InputGroup("PDB结构文件", "PDB文件 (*.pdb);;所有文件 (*.*)")
        input_grid.addWidget(pdb_group, 0, 1)
        
        # ESM特征文件
        esm_group = InputGroup("ESM特征文件", "特征文件 (*.pt *.npy);;所有文件 (*.*)")
        input_grid.addWidget(esm_group, 1, 1)
        
        # 模型权重文件
        model_group = InputGroup("模型权重文件", "模型文件 (*.pth *.pt *.h5);;所有文件 (*.*)")
        input_grid.addWidget(model_group, 2, 1)
        
        top_layout.addLayout(input_grid)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.predict_btn.setStyleSheet("background-color: #389138;")
        self.predict_btn.clicked.connect(self.start_prediction)
        
        self.stop_btn = QPushButton("停止预测")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("background-color: #D84315;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_prediction)
        
        self.export_btn = QPushButton("导出结果")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setStyleSheet("background-color: #0078D7;")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        
        control_layout.addWidget(self.predict_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.export_btn)
        control_layout.addStretch(1)
        
        top_layout.addLayout(control_layout)
        
        # 底部面板 - 结果和日志
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setSpacing(10)
        
        # 结果分隔器
        result_splitter = QSplitter(Qt.Horizontal)
        
        # 序列可视化区域
        sequence_viz_group = QGroupBox("序列结合位点可视化")
        sequence_viz_layout = QVBoxLayout()
        self.sequence_display = QTextBrowser()
        self.sequence_display.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 12pt;")
        sequence_viz_layout.addWidget(self.sequence_display)
        sequence_viz_group.setLayout(sequence_viz_layout)
        sequence_viz_group.setMinimumWidth(400)
        
        # 位点详情表格
        sites_table_group = QGroupBox("结合位点详情")
        sites_table_layout = QVBoxLayout()
        self.sites_table = QTableWidget()
        self.sites_table.setColumnCount(4)
        self.sites_table.setHorizontalHeaderLabels(["位置", "氨基酸", "置信度", "功能注释"])
        self.sites_table.horizontalHeader().setStretchLastSection(True)
        sites_table_layout.addWidget(self.sites_table)
        sites_table_group.setLayout(sites_table_layout)
        
        result_splitter.addWidget(sequence_viz_group)
        result_splitter.addWidget(sites_table_group)
        
        bottom_layout.addWidget(result_splitter, 4)
        
        # 进度条和日志区域
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("进度:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("准备就绪")
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #389138; }")
        progress_layout.addWidget(self.progress_bar, 1)
        
        bottom_layout.addLayout(progress_layout)
        
        # 日志区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(100)
        
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        
        bottom_layout.addWidget(log_group, 1)
        
        # 添加面板到分隔器
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        splitter.setSizes([400, 400])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
        # 保存输入组引用
        self.sequence_group = sequence_group
        self.pdb_group = pdb_group
        self.esm_group = esm_group
        self.model_group = model_group
        
        # 示例序列
        self.sequence_group.text_input.setPlainText(
            "MASTIGGKKKKVVEKQEAVQETGFSVEEDFEFDDEDDEDEDEDEEDPTPPTPTPTPEE"
            "SPTSEEEEEEEEGVQKQPPSAPPPATPAPQPATPAPQPATPAPAPTPEPAPAPQPSQE"
            "PATPAAPEVPPATPEEVQKQPATPKQPAPPETPQTPPAPPETPQTPEEEDEDEDEDED"
        )
    
    def start_prediction(self):
        # 获取输入值
        sequence = self.sequence_group.get_text()
        pdb_path = self.pdb_group.get_path()
        esm_path = self.esm_group.get_path()
        model_path = self.model_group.get_path()
        
        # 基本验证
        if not sequence:
            self.log_display.append("错误: 必须提供蛋白质序列!")
            self.progress_bar.setFormat("输入无效")
            return
            
        # 准备界面
        self.clear_results()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("开始预测...")
        self.log_display.clear()
        
        # 记录输入信息
        self.log_display.append("预测参数:")
        self.log_display.append(f" - 序列长度: {len(sequence)} 个氨基酸")
        if pdb_path: self.log_display.append(f" - PDB文件: {pdb_path}")
        if esm_path: self.log_display.append(f" - ESM特征: {esm_path}")
        if model_path: self.log_display.append(f" - 模型权重: {model_path}")
        self.log_display.append("-" * 80)
        
        # 创建并启动预测线程
        self.predictor = BindingSitePredictor(sequence, pdb_path, esm_path, model_path)
        self.predictor.update_progress.connect(self.update_progress)
        self.predictor.prediction_complete.connect(self.prediction_completed)
        
        self.predictor.start()
        self.predict_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def stop_prediction(self):
        if self.predictor and self.predictor.isRunning():
            self.predictor.stop()
            self.log_display.append("预测已中止")
            self.progress_bar.setFormat("预测中断")
            self.predict_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{progress}% - {message[:30]}{'...' if len(message)>30 else ''}")
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum())
    
    def prediction_completed(self, status, result):
        self.log_display.append(f"状态: {status}")
        self.log_display.append("-" * 80)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("预测完成!")
        
        # 保存结果
        self.prediction_result = result
        
        # 可视化结果
        self.visualize_sequence(result)
        self.populate_sites_table(result)
        
        # 启用导出按钮
        self.export_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def clear_results(self):
        self.sequence_display.clear()
        self.sites_table.setRowCount(0)
        self.prediction_result = None
        self.export_btn.setEnabled(False)
    
    def visualize_sequence(self, result):
        sequence = result["sequence"]
        binding_sites = result["binding_sites"]
        
        # 每行显示60个字符
        chunk_size = 60
        
        # 位置计数器
        position = 0
        line_number = 1
        
        # 清空显示
        self.sequence_display.clear()
        
        # 创建格式
        normal_format = QTextCharFormat()
        normal_format.setForeground(QBrush(QColor("#D4D4D4")))
        
        site_format = QTextCharFormat()
        site_format.setBackground(QBrush(QColor("#389138")))
        site_format.setForeground(QBrush(QColor("#FFFFFF")))
        
        # 添加位置标尺
        ruler = "       1         2         3         4         5         6\n"
        ruler += "       012345678901234567890123456789012345678901234567890123456789\n\n"
        self.sequence_display.setCurrentCharFormat(normal_format)
        self.sequence_display.append(ruler)
        
        # 处理序列
        for i in range(0, len(sequence), chunk_size):
            chunk = sequence[i:i+chunk_size]
            
            # 添加行号
            self.sequence_display.setCurrentCharFormat(normal_format)
            self.sequence_display.insertPlainText(f"{line_number:>3}  ")
            
            # 处理当前行
            for j, aa in enumerate(chunk):
                curr_pos = position + j
                
                if curr_pos in binding_sites:
                    self.sequence_display.setCurrentCharFormat(site_format)
                    self.sequence_display.insertPlainText(aa)
                else:
                    self.sequence_display.setCurrentCharFormat(normal_format)
                    self.sequence_display.insertPlainText(aa)
                    
            self.sequence_display.append("")
            position += chunk_size
            line_number += 1
        
        # 添加图例
        self.sequence_display.append("\n图例: 绿色背景表示预测的结合位点氨基酸")
    
    def populate_sites_table(self, result):
        binding_sites = result["binding_sites"]
        position_notes = result.get("position_notes", {})
        sequence = result["sequence"]
        
        # 排序位置
        sorted_positions = sorted(binding_sites.keys())
        
        # 准备表格
        self.sites_table.setRowCount(len(sorted_positions))
        
        for row, pos in enumerate(sorted_positions):
            # 位置
            pos_item = QTableWidgetItem(str(pos + 1))
            pos_item.setTextAlignment(Qt.AlignCenter)
            
            # 氨基酸
            if 0 <= pos < len(sequence):
                aa_item = QTableWidgetItem(sequence[pos])
            else:
                aa_item = QTableWidgetItem("?")
            aa_item.setTextAlignment(Qt.AlignCenter)
            
            # 置信度
            conf_item = QTableWidgetItem(f"{binding_sites[pos]:.2f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            
            # 功能注释
            note_item = QTableWidgetItem(position_notes.get(pos, "N/A"))
            
            # 添加到表格
            self.sites_table.setItem(row, 0, pos_item)
            self.sites_table.setItem(row, 1, aa_item)
            self.sites_table.setItem(row, 2, conf_item)
            self.sites_table.setItem(row, 3, note_item)
        
        # 调整列宽
        self.sites_table.resizeColumnsToContents()
    
    def export_results(self):
        if not self.prediction_result:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出预测结果", "", "文本文件 (*.txt);;CSV文件 (*.csv)"
        )
        
        if file_path:
            try:
                # 简单实现：实际应用中应实现更完整的导出
                with open(file_path, 'w') as f:
                    f.write("蛋白质结合位点预测结果\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # 输出序列和位点
                    sequence = self.prediction_result["sequence"]
                    binding_sites = self.prediction_result["binding_sites"]
                    
                    f.write(f"序列 (长度: {len(sequence)}):\n")
                    
                    # 每行显示60个字符
                    chunk_size = 60
                    for i in range(0, len(sequence), chunk_size):
                        chunk = sequence[i:i+chunk_size]
                        f.write(chunk + "\n")
                    
                    # 输出位置列表
                    f.write("\n预测的结合位点:\n")
                    f.write("位置\t氨基酸\t置信度\t注释\n")
                    
                    for pos, conf in sorted(binding_sites.items()):
                        if 0 <= pos < len(sequence):
                            aa = sequence[pos]
                        else:
                            aa = "?"
                        note = self.prediction_result.get("position_notes", {}).get(pos, "N/A")
                        f.write(f"{pos+1}\t{aa}\t{conf:.2f}\t{note}\n")
                    
                self.log_display.append(f"结果已导出到: {file_path}")
            except Exception as e:
                self.log_display.append(f"导出失败: {str(e)}")

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
    palette.setColor(QPalette.Highlight, QColor(56, 145, 56))  # 主题绿色
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = BindingSitePredictorApp()
    window.show()
    sys.exit(app.exec_())