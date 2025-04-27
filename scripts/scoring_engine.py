import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the parsed dataset
dataset_path = r'D:\cv-matcher\data\parsed_UpdatedResumeDataSet.csv'
df = pd.read_csv(dataset_path)

# Load a powerful pre-trained Sentence-BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # or use all-mpnet-base-v2 for better accuracy

# Function to clean and preprocess text data
def clean_text(text):
    # Remove special characters, extra spaces, and unwanted symbols
    text = text.strip().replace("\r\n", " ").replace("\n", " ")
    text = ' '.join(text.split())  # remove extra spaces
    return text

# Function to classify match scores into levels
def categorize_match(score):
    if score >= 0.50:
        return 'High'
    elif score >= 0.30:
        return 'Medium'
    elif score >= 0.10:
        return 'Low'
    else:
        return 'Very Low'

# Function to compute semantic similarity (cosine similarity)
def compute_semantic_similarity(job_description, resume_text):
    # Compute embeddings for job description and resume text
    job_desc_embedding = model.encode([job_description], convert_to_tensor=True)
    resume_embedding = model.encode([resume_text], convert_to_tensor=True)
    
    # Compute cosine similarity between the embeddings
    similarity_score = util.cos_sim(job_desc_embedding, resume_embedding)[0].cpu().numpy()[0]
    
    return similarity_score

# Function to compute match scores efficiently in batch
def compute_match_scores(df, job_description):
    # Clean and combine relevant parsed fields into one text
    combined_texts = df.apply(
        lambda row: ' '.join(filter(None, [
            clean_text(str(row.get('Parsed_Skills', ''))),
            clean_text(str(row.get('Parsed_Education', ''))),
            clean_text(str(row.get('Parsed_Experience', '')))
        ])),
        axis=1
    )

    # Generate embeddings for all resumes and the job description
    resume_embeddings = model.encode(combined_texts.tolist(), convert_to_tensor=True, show_progress_bar=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    # Compute similarity scores (cosine similarity)
    similarity_scores = util.cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()

    # Add scores and match levels to the DataFrame
    df['Match_Score'] = similarity_scores
    df['Match_Level'] = df['Match_Score'].apply(categorize_match)

    return df

# Function to calculate field-based scores (skills, experience, education)
def compute_field_scores(df, job_description):
    skills_score = []
    experience_score = []
    education_score = []

    # Loop over each CV and compute individual field scores
    for index, row in df.iterrows():
        # Calculate skill match score
        skills = clean_text(str(row.get('Parsed_Skills', '')))
        skills_score.append(compute_semantic_similarity(job_description, skills))

        # Calculate experience match score
        experience = clean_text(str(row.get('Parsed_Experience', '')))
        experience_score.append(compute_semantic_similarity(job_description, experience))

        # Calculate education match score
        education = clean_text(str(row.get('Parsed_Education', '')))
        education_score.append(compute_semantic_similarity(job_description, education))

    # Add these scores to the DataFrame
    df['Skills_Score'] = skills_score
    df['Experience_Score'] = experience_score
    df['Education_Score'] = education_score

    return df

# Function to compute final score considering weights
def compute_final_score(df, skill_weight=0.4, experience_weight=0.3, education_weight=0.3):
    # Final score based on weighted scores of individual fields
    df['Final_Score'] = (
        df['Skills_Score'] * skill_weight + 
        df['Experience_Score'] * experience_weight + 
        df['Education_Score'] * education_weight
    )
    return df

# Example job description
job_description = """
We are looking for a Data Scientist with experience in Python, machine learning, and data analysis.
The candidate should have a strong understanding of statistics and be comfortable with SQL and cloud platforms.
Experience with deep learning frameworks such as TensorFlow or PyTorch is preferred.
"""

# Run matching
df_matched = compute_match_scores(df, job_description)
df_matched = compute_field_scores(df_matched, job_description)
df_matched = compute_final_score(df_matched)

# Sort results and display top 5
top_n = 5
df_sorted = df_matched.sort_values(by='Final_Score', ascending=False)
print(f"\nTop {top_n} CVs with Final Scores:")
print(df_sorted[['Resume', 'Final_Score', 'Skills_Score', 'Experience_Score', 'Education_Score', 'Match_Level']].head(top_n))

# Save full results
matched_dataset_path = r'D:\cv-matcher\data\matched_UpdatedResumeDataSet.csv'
df_matched.to_csv(matched_dataset_path, index=False)
print(f"\n✅ Matched dataset saved to: {matched_dataset_path}")
