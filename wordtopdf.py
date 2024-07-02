import os
import re
from PyPDF2 import PdfReader
import pandas as pd

# Function to extract the number of languages from a resume text
def extract_languages(resume_text):
    matches = re.findall(r'Languages\s*Known\s*:\s*([a-zA-Z, ]+)', resume_text)
    if matches:
        languages = matches[0].split(',')
        return len(languages)
    else:
        return 0

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF file '{pdf_path}': {e}")
        return ''

# Function to process all resumes in a directory
def process_resumes(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):  # Assuming all CVs are in PDF format
            pdf_path = os.path.join(directory, filename)
            resume_text = extract_text_from_pdf(pdf_path)
            languages_count = extract_languages(resume_text)
            results.append({'Filename': filename, 'Languages Known': languages_count})
    return results

# Directory containing the CVs
cv_directory = r"C:\Users\Rohit Nigam\OneDrive\Documents\piramil\CV test"


# Process CVs and store the results in a DataFrame
results = process_resumes(cv_directory)
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_csv = "languages_known.csv"
df.to_csv(output_csv, index=False)

print("Languages known extracted and saved to", output_csv)