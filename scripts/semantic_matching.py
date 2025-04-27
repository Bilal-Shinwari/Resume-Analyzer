import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# === Load Dataset ===
dataset_path = r'D:\cv-matcher\data\parsed_UpdatedResumeDataSet.csv'
df = pd.read_csv(dataset_path)

# === Load Improved Sentence-BERT Model ===
# Higher accuracy than MiniLM
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 🔥 Highly recommended

# === Match Level Categorizer ===
def categorize_match(score):
    if score >= 0.50:
        return 'High'
    elif score >= 0.30:
        return 'Medium'
    elif score >= 0.10:
        return 'Low'
    else:
        return 'Very Low'

# === Main Matching Function ===
def compute_match_scores(df, job_description):
    print("\n🔄 Computing semantic embeddings...")

    # Combine parsed fields into a single string per resume
    combined_texts = df.apply(
        lambda row: ' '.join(filter(None, [
            str(row.get('Parsed_Skills', '')),
            str(row.get('Parsed_Education', '')),
            str(row.get('Parsed_Experience', ''))
        ])),
        axis=1
    )

    # Encode resumes and job description
    resume_embeddings = model.encode(
        combined_texts.tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    job_embedding = model.encode(
        job_description,
        convert_to_tensor=True
    )

    # Compute cosine similarities
    similarity_scores = util.cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()

    # Add scores and labels to the DataFrame
    df['Match_Score'] = similarity_scores
    df['Match_Level'] = df['Match_Score'].apply(categorize_match)

    return df

# === Job Description Example ===
job_description = """
We are looking for a Data Scientist with experience in Python, machine learning, and data analysis.
The candidate should have a strong understanding of statistics and be comfortable with SQL and cloud platforms.
Experience with deep learning frameworks such as TensorFlow or PyTorch is preferred.
"""

# === Run Matching ===
df_matched = compute_match_scores(df, job_description)

# === Show Top 5 Matches ===
top_n = 5
df_sorted = df_matched.sort_values(by='Match_Score', ascending=False)
print(f"\n📄 Top {top_n} CVs with Match Scores:")
print(df_sorted[['Resume', 'Match_Score', 'Match_Level']].head(top_n))

# === Save Results ===
matched_dataset_path = r'D:\cv-matcher\data\matched_UpdatedResumeDataSet.csv'
df_matched.to_csv(matched_dataset_path, index=False)
print(f"\n✅ Matched dataset saved to: {matched_dataset_path}")
