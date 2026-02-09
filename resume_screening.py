import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

def main():
    print("Loading Kaggle dataset...")
    try:
        # Load the resume dataset
        df = pd.read_csv('data/Resume.csv')
        # Clean category names
        df['Category'] = df['Category'].str.strip()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\nAvailable Job Categories:")
    categories = sorted(df['Category'].unique())
    for i, cat in enumerate(categories):
        print(f"{i+1}. {cat}")

    # Selected Job Role
    selected_role = "INFORMATION-TECHNOLOGY"
    print(f"\nSelected Job Role: {selected_role}")

    # Load resumes from resumes/ folder
    print("\nLoading resumes from 'resumes/' folder...")
    resume_files = glob.glob('resumes/*.txt')
    resumes_content = []
    resume_names = []
    
    for file_path in resume_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            resumes_content.append(content)
            resume_names.append(os.path.basename(file_path))
    
    num_resumes = len(resumes_content)
    print(f"Number of resumes loaded: {num_resumes}")

    if num_resumes == 0:
        print("No resumes found in 'resumes/' folder.")
        return

    # Ranking using TF-IDF and Cosine Similarity
    role_resumes = df[df['Category'] == selected_role]['Resume_str'].tolist()
    
    # Combine role resumes from dataset and new resumes to vectorize
    all_texts = role_resumes + resumes_content
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Separate vectors
    role_vectors = tfidf_matrix[:len(role_resumes)]
    input_vectors = tfidf_matrix[len(role_resumes):]
    
    # Create a profile for the job role
    role_profile = np.asarray(role_vectors.mean(axis=0))
    
    # Calculate cosine similarity
    similarities = cosine_similarity(role_profile, input_vectors)
    
    print("\nResume Ranking Results using TF-IDF and Cosine Similarity:")
    rankings = []
    for i in range(num_resumes):
        rankings.append((resume_names[i], similarities[0][i]))
    
    # Sort by similarity score descending
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(rankings):
        print(f"Rank {i+1}: {name} - Similarity Score: {score:.4f}")

    print("\nResume screening task completed successfully.")

if __name__ == "__main__":
    main()