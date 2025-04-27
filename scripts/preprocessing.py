import pandas as pd
import re
import spacy

# Load the dataset (adjust the file path accordingly)
dataset_path = r'D:\cv-matcher\data\UpdatedResumeDataSet.csv'  # Raw dataset path
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset Preview:")
print(df.head())

# Initialize the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Function to clean the CV text
def clean_text(text):
    # Remove unwanted characters and normalize text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    return text

# Function to extract entities using SpaCy (for example, skills, education, work experience)
def extract_entities(text):
    doc = nlp(text)
    # Extract named entities (for skills, education, etc.)
    skills = []
    for ent in doc.ents:
        if ent.label_ == 'ORG' or ent.label_ == 'GPE':  # Example: Capture skills or organizations (you can refine it)
            skills.append(ent.text)
    return skills

# Clean the 'Resume' column
df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

# Handle missing data - remove rows where 'Resume' column is empty
print("\nMissing Values Before Handling:")
print(df.isnull().sum())
df = df.dropna(subset=['Resume'])

# Alternatively, you could fill missing values in specific columns (for example, education or skills)
# df['Skills'] = df['Skills'].fillna('Not Available')

# Extract skills (for example) from the cleaned resume
df['Extracted_Skills'] = df['Cleaned_Resume'].apply(extract_entities)

# Display the cleaned dataset preview
print("\nCleaned Dataset Preview:")
print(df[['Resume', 'Cleaned_Resume', 'Extracted_Skills']].head())

# Save the cleaned dataset to a new CSV file in the 'data/' folder
cleaned_dataset_path = r'D:\cv-matcher\data\cleaned_UpdatedResumeDataSet.csv'
df.to_csv(cleaned_dataset_path, index=False)

print(f"\nCleaned dataset saved to: {cleaned_dataset_path}")
