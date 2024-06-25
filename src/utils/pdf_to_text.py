import os

from PyPDF2 import PdfReader


def extract_texts_from_folder(folder_path: str):
    """
    Extracts text from all PDF files in a folder, concatenates the results,
    and separates the text of each PDF with the name of the PDF file.

    Parameters:
    folder_path (str): The path to the folder containing PDF files.

    Returns:
    str: The concatenated extracted text from all PDFs, with each PDF's text
         separated by its file name.
    """
    all_texts = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = ""
            try:
                with open(file_path, "rb") as file:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text()
            except Exception as e:
                print(f"An error occurred while processing {file_path}: {e}")
                continue

            all_texts += text

    return all_texts
