import os
import glob
import pdfplumber
import win32com.client
from config import DATA_PATH, PDF_PATH

def extract_text_from_hwp(file_path: str) -> str:
    text = ""
    try:
        hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
        hwp.Open(file_path)
        hwp.XHwpWindows.Item(0).Visible = False

        text = hwp.GetTextFile("TEXT", "UTF-8")  
        if text is None:
            text = "" 
        hwp.Quit()
    except Exception as e:
        print(f"Error extracting text from HWP with win32com: {e}")
    return text

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

if __name__ == "__main__":
    hwp_files = glob.glob(os.path.join(DATA_PATH, "files", "*.hwp"))
    pdf_output_dir = PDF_PATH
    os.makedirs(pdf_output_dir, exist_ok=True)

    for hwp_file in hwp_files:
        base_name = os.path.basename(hwp_file)
        pdf_file_name = os.path.splitext(base_name)[0] + ".pdf"
        pdf_file_path = os.path.join(pdf_output_dir, pdf_file_name)

        if os.path.exists(pdf_file_path):
            text = extract_text_from_pdf(pdf_file_path)
            print(f"Extracted text from {pdf_file_name}:\n{text[:500]}...\n")
        else:
            print(f"PDF conversion failed for {hwp_file}")
        print("End of processing one file.\n")
        break  # Remove this break to process all files

