import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar,
    QGroupBox, QFormLayout, QLineEdit, QSplitter, QMessageBox,
    QStatusBar, QGridLayout, QCheckBox, QStackedWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon,QPixmap,QPainter,QPen
from train_app import ProteinTrainingApp

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DiConSite - Protein Binding Site Prediction")
        self.setGeometry(100, 100, 1200, 800)

        # self.train_btn.clicked.connect(self.open_training_app)
        # self.predict_btn.clicked.connect(self.open_prediction_app)
        
        # Apply styles
        self.apply_styles()
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("DiConSite - Protein Binding Site Prediction")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #61AFEF;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("A unified deep learning framework for precise prediction of multiple types of molecular binding sites")
        desc_label.setFont(QFont("Arial", 14))
        desc_label.setStyleSheet("color: #E0E0E0;")
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # Create splitter for content
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - software introduction
        intro_panel = QWidget()
        intro_layout = QVBoxLayout(intro_panel)
        intro_layout.setContentsMargins(20, 20, 20, 20)
        
        # Software introduction title
        intro_title = QLabel("Software Introduction")
        intro_title.setFont(QFont("Arial", 16, QFont.Bold))
        intro_title.setStyleSheet("color: #61AFEF; margin-bottom: 15px;")
        intro_layout.addWidget(intro_title)
        
        # Software introduction text
        intro_text = QTextEdit()
        intro_text.setReadOnly(True)
        intro_text.setFont(QFont("Arial", 12))
        intro_text.setStyleSheet("background-color: #252526; border: none; padding: 10px;")
        intro_text.setHtml("""
            <p style="line-height: 1.5; text-align: justify;">
                We propose <b>DiConSite</b>, a unified deep learning framework based on a contrastive learning-enhanced 
                self-distillation strategy, achieving precise prediction of multiple types of molecular binding sites 
                through the integration of E(n)-equivariant graph neural networks (EGNN). DiConSite combines semantic 
                representations from protein language models (PLM) with geometric features of protein structures via EGNN, 
                enabling effective structure-aware modeling.
            </p>
            <p style="line-height: 1.5; text-align: justify; margin-top: 15px;">
                At the model architecture level, we design a function-driven contrastive learning self-distillation 
                structure that, through latent space similarity constraints, encourages similar binding sites to form 
                compact clusters, enhancing the representational discriminability of the teacher model. Additionally, 
                the deep teacher network supervises the shallow student network, improving the model's generalization 
                ability and robustness.
            </p>
            <p style="line-height: 1.5; text-align: justify; margin-top: 15px;">
                Experimental results show that DiConSite outperforms the latest methods in predicting protein, peptide, 
                DNA, and RNA binding sites.
            </p>
        """)
        intro_layout.addWidget(intro_text)
        
        # Key features
        features_title = QLabel("Key Features")
        features_title.setFont(QFont("Arial", 14, QFont.Bold))
        features_title.setStyleSheet("color: #61AFEF; margin-top: 20px; margin-bottom: 10px;")
        intro_layout.addWidget(features_title)
        
        features_layout = QGridLayout()
        features_layout.setHorizontalSpacing(20)
        features_layout.setVerticalSpacing(10)
        
        features = [
            ("EGNN Integration", "Incorporates E(n)-equivariant graph neural networks for geometric feature extraction"),
            ("Contrastive Learning", "Uses contrastive learning to enhance representation discriminability"),
            ("Self-Distillation", "Implements self-distillation strategy for improved generalization"),
            ("Multi-Type Prediction", "Predicts protein, peptide, DNA, and RNA binding sites"),
            ("Structure-Aware Modeling", "Combines semantic and geometric features for comprehensive analysis"),
            ("Superior Performance", "Outperforms state-of-the-art methods in experimental evaluations")
        ]
        
        for i, (title, desc) in enumerate(features):
            row = i // 2
            col = (i % 2) * 2
            
            # Feature icon (placeholder)
            icon_label = QLabel()
            icon_label.setFixedSize(32, 32)
            icon_label.setStyleSheet("background-color: #0078D7; border-radius: 16px;")
            features_layout.addWidget(icon_label, row, col)
            
            # Feature text
            feature_layout = QVBoxLayout()
            feature_title = QLabel(title)
            feature_title.setFont(QFont("Arial", 11, QFont.Bold))
            feature_title.setStyleSheet("color: #E0E0E0;")
            
            feature_desc = QLabel(desc)
            feature_desc.setFont(QFont("Arial", 10))
            feature_desc.setStyleSheet("color: #A0A0A0;")
            feature_desc.setWordWrap(True)
            
            feature_layout.addWidget(feature_title)
            feature_layout.addWidget(feature_desc)
            features_layout.addLayout(feature_layout, row, col+1)
        
        intro_layout.addLayout(features_layout)
        
        # Right panel - architecture image
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(20, 20, 20, 20)
        
        # Image title
        image_title = QLabel("DiConSite Architecture")
        image_title.setFont(QFont("Arial", 16, QFont.Bold))
        image_title.setStyleSheet("color: #61AFEF; margin-bottom: 15px;")
        image_layout.addWidget(image_title)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1E1E1E; border: 1px solid #3A3A3F;")
        self.image_label.setMinimumSize(600, 400)
        
        # Try to load image
        try:
            # Use a placeholder image path - in a real application, this would be the actual image path
            image_path = "/Users/shawn/lqszchen/Project/PPIS/GUI_pyqt5/figs/F1.png"
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.create_placeholder_image()
        except:
            self.create_placeholder_image()
        
        image_layout.addWidget(self.image_label)
        
        # Image description
        image_desc = QTextEdit()
        image_desc.setReadOnly(True)
        image_desc.setFont(QFont("Arial", 10))
        image_desc.setStyleSheet("background-color: #252526; border: none; padding: 10px;")
        image_desc.setHtml("""
            <p style="line-height: 1.5; text-align: justify;">
                <b>DiConSite Architecture Overview:</b> The framework integrates protein language models (ESM) with 
                geometric features using E(n)-equivariant graph neural networks (EGNN). The architecture employs a 
                contrastive learning-enhanced self-distillation strategy with multiple EGNN layers. The shallow student 
                model learns from the deepest teacher model through knowledge distillation, while contrastive learning 
                enhances representation discriminability.
            </p>
            <p style="line-height: 1.5; text-align: justify; margin-top: 10px;">
                The model processes protein sequences and structures to predict various binding sites (P-PBs, P-RBs, 
                P-DBs, P-PepBs) with high accuracy.
            </p>
        """)
        image_layout.addWidget(image_desc)
        
        # Add panels to splitter
        content_splitter.addWidget(intro_panel)
        content_splitter.addWidget(image_panel)
        content_splitter.setSizes([500, 500])
        
        main_layout.addWidget(content_splitter, 1)
        
        # Button group
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        button_layout.setContentsMargins(50, 20, 50, 20)
        
        # Training button
        self.train_btn = QPushButton("Protein Binding Site Training")
        self.train_btn.setMinimumHeight(60)
        self.train_btn.setFont(QFont("Arial", 14))
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                border-radius: 8px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #0095FF;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
        """)
        self.train_btn.setToolTip("Train protein binding site prediction models")
        button_layout.addWidget(self.train_btn)
        
        # Prediction button (placeholder)
        self.predict_btn = QPushButton("Protein Binding Site Prediction")
        self.predict_btn.setMinimumHeight(60)
        self.predict_btn.setFont(QFont("Arial", 14))
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border-radius: 8px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """)
        self.predict_btn.setToolTip("Predict protein binding sites using trained models")
        button_layout.addWidget(self.predict_btn)
        
        main_layout.addLayout(button_layout)
        
        # Footer
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(10, 10, 10, 10)
        
        version_label = QLabel("DiConSite v1.0 | © 2025 Protein Analysis Suite")
        version_label.setFont(QFont("Arial", 10))
        version_label.setStyleSheet("color: #888888;")
        
        footer_layout.addWidget(version_label)
        footer_layout.addStretch()
        
        # Add citation
        citation_label = QLabel("Citation: [Your Paper Title Here], Journal, 2025")
        citation_label.setFont(QFont("Arial", 10))
        citation_label.setStyleSheet("color: #888888;")
        footer_layout.addWidget(citation_label)
        
        main_layout.addLayout(footer_layout)
    
    # def open_training_app(self):
    #     self.training_app = ProteinTrainingApp()
    #     self.training_app.show()

    # def open_prediction_app(self):
    #     # self.prediction_app = ProteinPredictionApp()
    #     self.prediction_app.show()

    def create_placeholder_image(self):
        """Create a placeholder image with architecture description"""
        # Create a pixmap to draw on
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(30, 30, 30))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set font and colors
        title_font = QFont("Arial", 16, QFont.Bold)
        text_font = QFont("Arial", 10)
        painter.setFont(title_font)
        painter.setPen(QColor(97, 175, 239))
        
        # Draw title
        painter.drawText(20, 40, "DiConSite Architecture")
        
        # Draw description
        painter.setFont(text_font)
        painter.setPen(QColor(200, 200, 200))
        
        lines = [
            "Input: Protein sequence and structure features",
            "Processing: ESM embeddings and geometric features",
            "Model: EGNN layers with self-distillation",
            "Output: Binding site probabilities (P-PBs, P-RBs, etc.)",
            "",
            "Blue: Input features",
            "Green: Processing modules",
            "Red: Output predictions"
        ]
        
        for i, line in enumerate(lines):
            painter.drawText(40, 80 + i * 25, line)
        
        # Draw simple diagram elements
        # Input
        painter.setBrush(QColor(97, 175, 239))
        painter.drawRect(400, 80, 150, 40)
        painter.drawText(410, 105, "Input Features")
        
        # Processing
        painter.setBrush(QColor(76, 175, 80))
        painter.drawRect(400, 150, 150, 40)
        painter.drawText(430, 175, "EGNN Layers")
        
        # Output
        painter.setBrush(QColor(244, 67, 54))
        painter.drawRect(400, 220, 150, 40)
        painter.drawText(420, 245, "Binding Site Predictions")
        
        # Arrows
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(475, 120, 475, 150)  # Input to processing
        painter.drawLine(475, 190, 475, 220)  # Processing to output
        
        painter.end()
        
        self.image_label.setPixmap(pixmap)
    
    def apply_styles(self):
        """Apply stylesheet"""
        self.setStyleSheet("""
            QWidget {
                background-color: #2D2D30;
            }
            QLabel {
                color: #E0E0E0;
            }
            QTextEdit {
                background-color: #252526;
                color: #E0E0E0;
                border: 1px solid #3A3A3F;
                border-radius: 5px;
            }
            QPushButton {
                font-size: 14px;
                min-height: 40px;
                padding: 8px 16px;
                border-radius: 5px;
                background-color: #4A4A4F;
                color: white;
            }
            QPushButton:hover {
                background-color: #5A5A5F;
            }
            QPushButton:pressed {
                background-color: #3A3A3F;
            }
        """)

# Main application with stacked widgets
class ProteinAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Analysis Suite")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create stacked widget for multiple screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create main menu
        self.main_menu = MainMenu()
        self.stacked_widget.addWidget(self.main_menu)
        
        # Create training app (but don't add it yet)
        self.training_app = ProteinTrainingApp()
        self.training_app.close_signal.connect(self.show_main_menu)
        
        # Connect buttons
        self.main_menu.train_btn.clicked.connect(self.show_training_app)
        
        # Apply global styles
        self.apply_styles()
        
        # Show main menu initially
        self.show_main_menu()
    
    def apply_styles(self):
        """Apply global stylesheet"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
        """)
    
    def show_main_menu(self):
        """Show the main menu"""
        # Remove training app if it exists in stack
        if self.stacked_widget.indexOf(self.training_app) != -1:
            self.stacked_widget.removeWidget(self.training_app)
        
        # Reset training app
        self.training_app = ProteinTrainingApp()
        self.training_app.close_signal.connect(self.show_main_menu)
        
        # Show main menu
        self.stacked_widget.setCurrentWidget(self.main_menu)
    
    def show_training_app(self):
        """Show the training application"""
        # Add training app to stack if not already added
        if self.stacked_widget.indexOf(self.training_app) == -1:
            self.stacked_widget.addWidget(self.training_app)
        
        # Show training app
        self.stacked_widget.setCurrentWidget(self.training_app)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main menu
    window = ProteinAnalysisApp()
    window.show()
    
    sys.exit(app.exec_())