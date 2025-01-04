import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("모던 대시보드")
        self.setGeometry(100, 100, 800, 600)

        # 중앙 위젯과 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # UI 요소 추가
        label = QLabel("환영합니다! 여기에 대시보드 정보를 표시합니다.")
        button = QPushButton("데이터 새로 고침")

        layout.addWidget(label)
        layout.addWidget(button)

        central_widget.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())
