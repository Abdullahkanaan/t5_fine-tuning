import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import torch

class T5SummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.tokenizer = AutoTokenizer.from_pretrained("/home/encoder/Desktop/workspaces/LLMs_workspace/t5_small/results_t5small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("/home/encoder/Desktop/workspaces/LLMs_workspace/t5_small/results_t5small")


    def initUI(self):
        self.setWindowTitle("T5 Metin Özetleme")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        
        self.input_text = QTextEdit(self)
        self.input_text.setPlaceholderText("Özetlenecek metni buraya girin...")
        layout.addWidget(self.input_text)
        
        self.summarize_button = QPushButton("Özetle", self)
        self.summarize_button.clicked.connect(self.summarize_text)
        layout.addWidget(self.summarize_button)
        
        self.output_label = QLabel("Özet:")
        layout.addWidget(self.output_label)
        
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
        self.setLayout(layout)


    def summarize_text(self):
        input_text = self.input_text.toPlainText()
        if not input_text.strip():
            self.output_text.setText("Lütfen bir metin girin.")
            return

        # Model için giriş formatı
        input_ids = self.tokenizer("summarize: " + input_text, return_tensors="pt", max_length=2048, truncation=True).input_ids
        summary_ids = self.model.generate(input_ids, max_length=2048, min_length=8, length_penalty=2.0, num_beams=5, no_repeat_ngram_size=2, early_stopping=True) # 
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        self.output_text.setText(summary_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = T5SummarizerApp()
    window.show()
    sys.exit(app.exec_())
