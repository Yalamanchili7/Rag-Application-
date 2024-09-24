import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from climate_data import climate_data

# Set up the retriever
encoder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = encoder.encode(climate_data)

# Set up the generator
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def rag_pipeline(query, k=3):
    # Retrieve relevant documents
    query_embedding = encoder.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    retrieved_contexts = [climate_data[i] for i in top_k_indices]

    # Generate answer
    context = " ".join(retrieved_contexts)
    input_text = f"Answer the following question based on the given context:\nContext: {context}\nQuestion: {query}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

if __name__ == "__main__":
    while True:
        question = input("Enter your question about climate change (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = rag_pipeline(question)
        print(f"Answer: {answer}\n")
