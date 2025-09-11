# `Study Buddy Chatbot AI`

## Description
Chatbot edukatif yang dirancang untuk membantu pelajar memahami materi dengan interaktif, adaptif, dan menyenangkan.
Berfokus untuk mendukung Quality Education (SDG #4) melalui pembelajaran berbasis dialog, refleksi, dan personalisasi.

**Note for developers : fork the repo first before edit or commit.**

## 🧑‍💻 Team

|          **Name**          |      **Role**       |
|----------------------------|---------------------|
| Muhammad Alvin Ababil      | Project Manager     |
| Kemas M Aryadary Rasyad    | Front End Technical |
| Karina Azzahra             | Front End UI/UX     |
| M Rafli Adhan S            | Back End            |

## 🚀 Features
- **🤖 AI Powered Chatbot**                   : Bisa menjawab pertanyaan, menjelaskan materi dan memberi latihan soal. Didukung oleh model AI yang adaptif terhadap gaya belajar penguna.
- **📊 Adaptive Learning Flow**               : Sistem pembelajaran menyesuaikan performa dan respons pengguna.
- **📄 Multi-Format Document Support**        : Bisa memproses dokumen dengan format PDF, DOCX, Markdown, TXT, dan HTML.
- **💾 Vector Database**                    : Menggunakan chromaDB untuk pencarian data secara efisien.
- **🌐 Decentralized Frontend Architecture**  : Frontend bersifat modular dan scalable sehingga bisa mengembangkan fitur tanpa mengaggu sistem utama.

## 🛠 Tech Stack

**Frontend:**
- Streamlit
- HTML/CSS

**Backend:**
- Python 3.13 
- Google Gemini AI 
- Sentence Transformers 
- ChromaDB 
- FastAPI 
- Pydantic 
- PyPDF2
- python-docx
- BeautifulSoup4 
- Markdown
- tiktoken
- ChromaDB

## 🚀 How to Run the Project

### Step 1. Clone the Repository
``` bash
cd studybuddy

pip install -r requirements.txt

cp .env.example .env
```
### Step 2. Add Gemini API Key
Add to .env file:
```
GEMINI_API_KEY = insert gemini api key here
```
### Step 3 Run the Application
```
python run.py
```

## 📋 Requirements
- Python 3 or newer
- Gemini API Key

