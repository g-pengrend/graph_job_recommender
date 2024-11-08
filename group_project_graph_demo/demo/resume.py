import PyPDF2

def load_and_display_resume(pdf_path):
    print(f"Loading resume from {pdf_path}...")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    print("Resume text extracted successfully!")
    print(text)

load_and_display_resume('./demo/data/Profile.pdf')
