import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests  # For online resources (to integrate learning platforms later)

# Function to extract keywords from text (can use TF-IDF for more advanced analysis)
def extract_keywords(text, top_n=10):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the TF-IDF values for each word
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Create a dictionary of words with their corresponding TF-IDF score
    word_score = dict(zip(feature_names, tfidf_scores))
    
    # Sort words by TF-IDF score in descending order and get top N
    sorted_words = sorted(word_score.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, score in sorted_words[:top_n]]

# Function to find missing skills from the CV
def find_missing_skills(job_description, cv_text):
    # Extract keywords from the job description and CV
    job_keywords = extract_keywords(job_description)
    cv_keywords = extract_keywords(cv_text)
    
    # Find the missing skills (i.e., skills in the job description but not in the CV)
    missing_skills = list(set(job_keywords) - set(cv_keywords))
    
    return missing_skills

# Function to suggest learning resources for missing skills
def suggest_learning_resources(missing_skills):
    # For simplicity, we provide dummy links for missing skills
    recommendations = {}
    
    for skill in missing_skills:
        recommendations[skill] = f"https://www.coursera.org/search?query={skill}"  # Example search URL on Coursera
    
    return recommendations

# Example data: Job description and CV text
job_description = """
We are looking for a Data Scientist with experience in Python, machine learning, data analysis, and deep learning. 
Experience with TensorFlow, PyTorch, and SQL is required.
"""

cv_text = """
Skilled Data Scientist with experience in Python, machine learning, data analysis. Familiar with SQL.
"""

# Perform gap analysis
missing_skills = find_missing_skills(job_description, cv_text)

# Get learning recommendations
learning_resources = suggest_learning_resources(missing_skills)

# Output the missing skills and their learning resources
print("Missing Skills and Recommendations:")
for skill, resource in learning_resources.items():
    print(f"Skill: {skill} -> Learn more: {resource}")
