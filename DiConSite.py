import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget, QLabel, QHBoxLayout, QMessageBox, QAction, QFileDialog
from PyQt5.QtGui import QIcon, QFont, QPixmap, QCloseEvent, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
# from util import Modules, InputDialog, PlotWidgets, MachineLearning
from task import train1,prediction
import qdarkstyle
from qdarkstyle.light.palette import LightPalette
import threading
import pandas as pd
# from qt_material import apply_stylesheet

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        # initialize GUI
        self.setWindowTitle('DiConSite')
        # self.resize(1920, 1080)
        self.setMaximumSize(1920, 1080)
        self.setMinimumSize(1920, 1080)
        self.setWindowIcon(QIcon('/Users/shawn/lqszchen/Project/PPIS/GUI_pyqt5/figs/logo.png'))
        bar = self.menuBar()
        
        predict = bar.addMenu('Model Prediction')
        predict.setFont(QFont("Arial", 14))
        basic_predict = QAction('Prediction', self)
        basic_predict.triggered.connect(self.openLoadpredictWindow)
        quit = QAction('Exit', self)
        quit.triggered.connect(self.closeEvent)
        predict.addAction(basic_predict)
        predict.addSeparator()
        predict.addAction(quit)
        


        train = bar.addMenu('Model Training')
        train.setFont(QFont('Arial',14))
        basic_train = QAction('Training', self)
        basic_train.triggered.connect(self.openLoadtrainWindow)
        quit_train = QAction('Exit', self)
        quit_train.triggered.connect(self.closeEvent)
        train.addAction(basic_train)
        train.addSeparator()
        train.addAction(quit)
        bar.setFixedHeight(50)


        # move window to center
        self.moveCenter()

        self.widget = QWidget()
        hLayout = QHBoxLayout(self.widget)
        hLayout.setAlignment(Qt.AlignCenter)
        label = QLabel()
        # label.setMaximumWidth(600)
        label.setPixmap(QPixmap('/Users/shawn/lqszchen/Project/PPIS/GUI_pyqt5/figs/gui.png'))
        label.setScaledContents(True)
        hLayout.addWidget(label)
        self.setCentralWidget(self.widget)

    def moveCenter(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def openLoadtrainWindow(self):
        # app1 = QApplication(sys.argv)
        #
        # extra = {
        #
        #     # Density Scale
        #     'density_scale': '2',
        # }
        # apply_stylesheet(app1, theme='dark_cyan.xml', extra=extra)
        self.loadWin1 = train1.ProteinTrainingApp()
        # apply_stylesheet(self.loadWin, theme='dark_cyan.xml', extra=extra)
        # self.loadWin.setFont(QFont('Arial', 10))
        # self.loadWin.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
        self.loadWin1.close_signal.connect(self.recover)
        self.loadWin1.show()
        self.setDisabled(True)
        self.setVisible(False)

    def openLoadpredictWindow(self):
        # app = QApplication(sys.argv)
        #
        # extra = {
        #
        #     # Density Scale
        #     'density_scale': '2',
        # }
        # apply_stylesheet(app, theme='dark_cyan.xml', extra=extra)
        self.loadWin2 = prediction.BindingSitePredictorApp()
        # self.loadWin.setFont(QFont('Arial', 10))
        # self.loadWin.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
        # self.loadWin2.close_signal.connect(self.recover)
        self.loadWin2.show()
        self.setDisabled(True)
        self.setVisible(False)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit(0)
        else:
            if event:
                event.ignore()
    def recover(self, module):
        try:
            if module == 'Basic':
                del self.basicWin
            elif module == 'Estimator':
                del self.estimatorWin
            elif module == 'AutoML':
                del self.mlWin
            elif module == 'LoadModel1':
                del self.loadWin1
            elif module == 'LoadModel2':
                del self.loadWin2
            else:
                pass
        except Exception as e:
            pass
        self.setDisabled(False)
        self.setVisible(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # extra = {
    #
    #     # Density Scale
    #     'density_scale': '1',
    # }
    # apply_stylesheet(app, theme='dark_cyan.xml', extra=extra)
    window = MainWindow()

    window.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    # window.setStyleSheet(qdarkstyle.load_stylesheet_pyside())
    window.show()
    sys.exit(app.exec_())