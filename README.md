# Resume Screening using NLP (TF-IDF & Cosine Similarity)

## 📌 Project Overview
This project implements an **NLP-based Resume Screening System** that ranks resumes
based on their relevance to a selected job role.  
It uses **TF-IDF vectorization** and **Cosine Similarity** to compare resumes against
job-related resume data sourced from a Kaggle dataset.

The system helps automate the initial screening process by identifying
the most relevant resumes efficiently.

---------------------------------------------------------------------------------------------

## 🎯 Objective
- To automate resume screening using Natural Language Processing
- To rank resumes based on similarity to a selected job role
- To demonstrate practical usage of TF-IDF and cosine similarity in Machine Learning

---------

## 📂 Dataset
- **Source:** Kaggle Resume Dataset  
- **Description:** Contains resume text categorized by job roles such as
  Information Technology, HR, Finance, etc.
- The dataset includes HTML content and special characters, which are cleaned
  during preprocessing.

> Note: The dataset is used for educational and internship purposes only.

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- Natural Language Processing (NLP)

------------------------------------------------------------------------------------

## 🚀 How It Works
1. Loads the Kaggle resume dataset
2. Displays available job categories
3. Selects a job role automatically
4. Cleans and preprocesses resume text
5. Converts text into TF-IDF vectors
6. Calculates cosine similarity
7. Ranks resumes based on similarity scores
