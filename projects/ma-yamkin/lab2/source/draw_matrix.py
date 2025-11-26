import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import numpy as np
import os
import pickle


def visualize_term_document_matrix(td_matrix, term_to_index, output_path, max_terms=50, max_docs=50, figsize=(12, 8)):
    n_terms, n_docs = td_matrix.shape
    
    terms_subset = min(max_terms, n_terms)
    docs_subset = min(max_docs, n_docs)
    submatrix = td_matrix[:terms_subset, :docs_subset].toarray()
    
    terms = list(term_to_index.keys())[:terms_subset]
    doc_labels = [f"doc_{i:05d}" for i in range(docs_subset)]
    
    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid")
    
    sns.heatmap(
        submatrix,
        annot=False,
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "TF (term frequency)"},
        xticklabels=doc_labels,
        yticklabels=terms
    )
    
    plt.xlabel("Документ")
    plt.ylabel("Термин")
    plt.title(f"Матрица термин документ (Топ {terms_subset} терминов × {docs_subset} документов)")
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()  


if __name__ == "__main__":
    output_dir = os.path.join('..', 'assets', 'annotated-corpus', 'term_doc_matrix_train')

    with np.load(os.path.join(output_dir, 'term_document_matrix.npz')) as data:
        td_matrix = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])

    with open(os.path.join(output_dir, 'term_to_index.pkl'), 'rb') as f:
        term_to_index = pickle.load(f)

    with open(os.path.join(output_dir, 'documents_filtered.pkl'), 'rb') as f:
        documents = pickle.load(f)

    viz_output_path = os.path.join('tdm_heatmap.png')
    visualize_term_document_matrix(
        td_matrix=td_matrix,
        term_to_index=term_to_index,
        output_path=viz_output_path,
        max_terms=100,   
        max_docs=100   
    )