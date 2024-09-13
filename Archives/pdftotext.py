
import fitz
import os 
class PDFtoText():
    def __init__(self) -> None:
        pass

    def open_pdf(self, pdf):
        if os.path.exists(str(pdf)) or isinstance(pdf,bytes):
                self.pdf = fitz.open(pdf)
                self.page_count = self.pdf.page_count
                return self.pdf
                # self.pdf.close()
        else:
            raise ValueError(f"PDF path is incorrect", pdf)
    def extract_all_text(self, pdf):
         # Open the PDF file
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)
        
        all_text = ''

        # Iterate through all pages
        for page_number in range(self.page_count) :
            # Get the page
            page = self.pdf[page_number]

            # Extract text from the page
            text = page.get_text()
            all_text += text

            # Print or process the extracted text as needed
            # print(f"Page {page_number + 1}:\n{text}\n")
        return all_text
    
    def extract_all_text_page_wise(self, pdf):
         # Open the PDF file
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)
        
        all_text = []

        # Iterate through all pages
        for page_number in range(self.page_count) :
            # Get the page
            page = self.pdf[page_number]

            # Extract text from the page
            text = page.get_text()
            all_text.append(text)
        return all_text
    def extract_text_from_single_page(self,pdf, page_number):
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)
        if page_number -1> self.page_count:
             raise ValueError("Invlaid pagenumber")
        else:
             return self.pdf[page_number-1].get_text()
    def extract_text_from_interval(self,pdf,page_number, interval =1):
        if not pdf: return None
        self.pdf = self.open_pdf(pdf)
        text = ""
        if page_number > self.page_count:
            raise ValueError("Invlaid pagenumber")
        else:
            # Calculate the start and end pages
            start_page = max(0, page_number - interval)
            end_page = min(self.page_count - 1, page_number + interval)

            for page_number in range(start_page, end_page + 1):
                text += self.extract_text_from_single_page(pdf=pdf, page_number=page_number)
        return text

if __name__ == "__main__":
    pdf_to_text = PDFtoText()
    text = pdf_to_text.extract_all_text("0001166691-23-000024.pdf")
    print(text)