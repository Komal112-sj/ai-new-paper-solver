from pypdf import PdfReader


def extract_text_from_pdf(pdf_file: str) -> str:
    """
    Extracts all text content from a PDF file.
    Handles empty pages safely.
    """

    reader = PdfReader(pdf_file)
    extracted_text = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(page_text)

    return "\n".join(extracted_text)
