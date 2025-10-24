#Personalized Course Recommendation Engine


import pandas as pd
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss

url = "https://raw.githubusercontent.com/Bluedata-Consulting/GAAPB01-training-code-base/refs/heads/main/Assignments/assignment2dataset.csv"
courses = pd.read_csv(url)

# Combine title and description into a single text field
courses['text'] = courses['title'].fillna("") + " " + courses['description'].fillna("")

# 3. Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(courses['text'].tolist(), convert_to_numpy=True)

# 4. Build FAISS index (cosine similarity)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 5. Recommendation function
def recommend_courses(profile: str, completed_ids: List[str], top_k: int = 10) -> List[Tuple[str, str, float]]:
    query_emb = model.encode([profile], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)
    
    recs = []
    for score, idx in zip(distances[0], indices[0]):
        cid = courses.iloc[idx]['course_id']
        if cid not in completed_ids:
            recs.append((cid, courses.iloc[idx]['title'], float(score)))
    return recs[:5]

# 6. Test Profiles
test_profiles = [
    ("I’ve completed the ‘Python Programming for Data Science’ course and enjoy data visualization.", ["C101"]),
    ("I know Azure basics and want to manage containers and build CI/CD pipelines.", []),
    ("My background is in ML fundamentals; I’d like to specialize in neural networks and production workflows.", []),
    ("I want to learn to build and deploy microservices with Kubernetes—what courses fit best?", []),
    ("I’m interested in blockchain and smart contracts but have no prior experience.", [])
]

# 7. Evaluation
for profile, completed in test_profiles:
    print("\n=== Profile:", profile)
    recs = recommend_courses(profile, completed)
    for cid, title, score in recs:
        print(f"   {cid} — {title} (score {score:.3f})")
