"""
Task 8: Main script to vectorize test set and save in TSV format.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# Add source to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.data_loader import load_corpus, get_sentences_as_tokens
from source.neural_vectorization import NeuralVectorizer
from source.document_vectorizer import DocumentVectorizer
from source.basic_vectorization import BasicVectorizer
from source.token_dictionary import TokenDictionary


def main():
    """
    Main function to vectorize test set and save results.
    """
    print("=" * 60)
    print("Lab 2 - Text Vectorization")
    print("Task 8: Vectorize test set and save in TSV format")
    print("=" * 60)
    
    # Configuration
    lab1_corpus_dir = "../lab1/assets/annotated_corpus"
    output_dir = "assets/embeddings"
    model_dir = "assets/models"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Step 1: Load training data to train the model
    print("\n[Step 1] Loading training data...")
    train_docs, train_labels, train_ids = load_corpus(lab1_corpus_dir, split='train')
    train_sentences = get_sentences_as_tokens(train_docs, use_lemma=True)
    
    # Flatten for Word2Vec training (list of all sentences)
    train_sentences_flat = []
    for doc in train_sentences:
        train_sentences_flat.extend(doc)
    
    print(f"Loaded {len(train_docs)} training documents")
    print(f"Total training sentences: {len(train_sentences_flat)}")
    
    # Step 2: Train neural vectorization model (Task 3)
    print("\n[Step 2] Training Word2Vec model...")
    neural_model = NeuralVectorizer(
        model_type='word2vec',
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=0  # CBOW
    )
    neural_model.train(train_sentences_flat, epochs=10)
    
    # Save model
    model_path = os.path.join(model_dir, "word2vec_model.model")
    neural_model.save(model_path)
    
    # Step 3: Load test data
    print("\n[Step 3] Loading test data...")
    test_docs, test_labels, test_ids = load_corpus(lab1_corpus_dir, split='test')
    test_sentences = get_sentences_as_tokens(test_docs, use_lemma=True)
    
    print(f"Loaded {len(test_docs)} test documents")
    
    # Step 4: Initialize document vectorizer (Task 7)
    print("\n[Step 4] Initializing document vectorizer...")
    doc_vectorizer = DocumentVectorizer(neural_model)
    
    # Step 5: Vectorize test documents
    print("\n[Step 5] Vectorizing test documents...")
    embeddings = []
    valid_doc_ids = []
    
    for doc_idx, (doc_sentences, doc_id) in enumerate(tqdm(zip(test_sentences, test_ids), 
                                                          total=len(test_docs),
                                                          desc="Vectorizing")):
        # Vectorize document using Task 7 method
        doc_vector = doc_vectorizer.vectorize_document_from_tokens(
            doc_sentences,
            use_tfidf_weights=False  # Set to True if you want TF-IDF weighted average
        )
        
        embeddings.append(doc_vector)
        valid_doc_ids.append(doc_id)
    
    embeddings = np.array(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Step 6: Save results in TSV format (Task 8)
    print("\n[Step 6] Saving embeddings to TSV file...")
    output_file = os.path.join(output_dir, "test_embeddings.tsv")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, embedding in zip(valid_doc_ids, embeddings):
            # Format: doc_id \t component1 \t component2 \t ... \t componentN
            embedding_str = '\t'.join([f"{val:.6f}" for val in embedding])
            f.write(f"{doc_id}\t{embedding_str}\n")
    
    print(f"Saved embeddings to {output_file}")
    print(f"Total documents: {len(valid_doc_ids)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    print("\n" + "=" * 60)
    print("Task 8 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
