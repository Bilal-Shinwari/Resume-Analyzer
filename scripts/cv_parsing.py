import pandas as pd
import spacy
import re

# Load the cleaned dataset
dataset_path = r'D:\cv-matcher\data\cleaned_UpdatedResumeDataSet.csv'  # Path to the cleaned dataset
df = pd.read_csv(dataset_path)

# Initialize the SpaCy model for NER
nlp = spacy.load('en_core_web_sm')

# Function to extract entities using SpaCy
def extract_entities(text):
    doc = nlp(text)
    # Initialize lists to store extracted entities
    skills = []
    education = []
    experience = []
    
    # Loop through entities found by SpaCy
    for ent in doc.ents:
        # For skills or organizations (custom rule, you can adjust this)
        if ent.label_ == 'ORG' or ent.label_ == 'PRODUCT':
            skills.append(ent.text)
        # For education (e.g., degree, university name)
        elif ent.label_ == 'ORG' or ent.label_ == 'GPE':  # Example rule for university
            education.append(ent.text)
        # For work experience (custom rule for years or positions)
        elif ent.label_ == 'DATE':
            experience.append(ent.text)
    
    return skills, education, experience

# Function to clean and extract structured data from each CV
def parse_cv_data(df):
    # Initialize lists to store parsed results
    parsed_skills = []
    parsed_education = []
    parsed_experience = []
    
    # Loop through each CV in the dataset
    for index, row in df.iterrows():
        text = row['Cleaned_Resume']  # Use the cleaned resume text
        
        # Extract structured data (skills, education, experience)
        skills, education, experience = extract_entities(text)
        
        # Append results to the respective lists
        parsed_skills.append(skills)
        parsed_education.append(education)
        parsed_experience.append(experience)
    
    # Add the parsed columns to the DataFrame
    df['Parsed_Skills'] = parsed_skills
    df['Parsed_Education'] = parsed_education
    df['Parsed_Experience'] = parsed_experience

    return df

# Parse the CVs and extract structured data
df_parsed = parse_cv_data(df)

# Display the parsed data preview
print("\nParsed Data Preview:")
print(df_parsed[['Resume', 'Parsed_Skills', 'Parsed_Education', 'Parsed_Experience']].head())

# Save the parsed dataset to a new CSV file
parsed_dataset_path = r'D:\cv-matcher\data\parsed_UpdatedResumeDataSet.csv'
df_parsed.to_csv(parsed_dataset_path, index=False)

print(f"\nParsed dataset saved to: {parsed_dataset_path}")
