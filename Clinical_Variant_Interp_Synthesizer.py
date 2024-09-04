import pandas as pd
import gzip
import re
import random
from transformers import AutoModel, AutoTokenizer
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Load and preprocess ClinVar data from .txt.gz files
def load_clinvar_data(conflicting_interps_file, variant_summary_file):
    """
    Load ClinVar data from the gzipped tab-delimited files. 
    Reads the conflicting_interpretations.txt.gz and variant_summary.txt.gz files using pandas.
    """
    with gzip.open(conflicting_interps_file, 'rt') as f:
        conflicting_interps = pd.read_csv(f, sep='\t', low_memory=False)
    with gzip.open(variant_summary_file, 'rt') as f:
        variant_summary = pd.read_csv(f, sep='\t', low_memory=False)
    return conflicting_interps, variant_summary

# Extract relevant PMIDs from free-text descriptions
def extract_pmids(description):
    """
    Extract PubMed IDs (PMIDs) from the free-text descriptions in the conflicting interpretations.
    Uses a regular expression to search for patterns like 'PMID: 12345678'.
    """
    pmid_pattern = r"PMID:?\s?(\d+)"
    return re.findall(pmid_pattern, description)

# Initialize BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Generate BioBERT embeddings
def generate_biobert_embeddings(text):
    """
    Generate BioBERT embeddings for a given text. 
    Converts the text into tokenized input and passes it through the BioBERT model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = biobert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Parse the question to extract gene and condition
def parse_question(question):
    """
    Parse the input question to extract the gene and condition being queried.
    Assumes a format like 'for GENE in CONDITION'.
    """
    gene_match = re.search(r"for (\w+ \w+)", question)
    condition_match = re.search(r"in (\w+ \w+)", question)
    gene = gene_match.group(1) if gene_match else "Unknown"
    condition = condition_match.group(1) if condition_match else "Unknown"
    return gene, condition

# Generate clinical summary for a variant
def synthesize_variant_summary(variant_data, conflicting_data, gene, condition):
    """
    Synthesize a clinical summary for a given variant. 
    Extracts relevant variant information, including clinical significance, conflicting interpretations, and PMIDs.
    Returns a summary based on the data for the given gene and condition.
    """
    variant_subset = variant_data[
        (variant_data['GeneSymbol'] == gene) &
        (variant_data['PhenotypeList'].str.contains(condition, na=False))
    ]
    conflicting_subset = conflicting_data[conflicting_data['Gene_Symbol'] == gene]
    
    summaries = []
    for index, variant in variant_subset.iterrows():
        clinical_significance = variant['ClinicalSignificance']
        review_status = variant['ReviewStatus']
        condition_list = variant['PhenotypeList']
        
        conflict_row = conflicting_subset[conflicting_subset['NCBI_Variation_ID'] == variant['VariationID']]
        if not conflict_row.empty:
            submitter1_sig = conflict_row.iloc[0].get('Submitter1_ClinSig', "Unknown")
            submitter2_sig = conflict_row.iloc[0].get('Submitter2_ClinSig', "Unknown")
            conflicting_significance = f"Submitter1: {submitter1_sig}, Submitter2: {submitter2_sig}"
        else:
            conflicting_significance = "No conflicting interpretations found."

        pmids = extract_pmids(conflict_row.iloc[0].get('Submitter1_Description', '')) + \
                extract_pmids(conflict_row.iloc[0].get('Submitter2_Description', ''))

        summary = f"Gene: {gene}\nCondition: {condition}\nClinical Significance: {clinical_significance}\n"
        summary += f"Review Status: {review_status}\nAssociated Conditions: {condition_list}\n"
        summary += f"Conflicting Clinical Interpretations: {conflicting_significance}\n"
        if pmids:
            summary += f"Relevant literature references (PMIDs): {', '.join(pmids)}\n"
        summary += "This synthesis is a preliminary interpretation. Please review and modify for accuracy.\n"
        summaries.append(summary)
    
    return summaries

# Create the ClinVar index for retrieval
def create_clinvar_index(variant_data):
    """
    Create a ClinVar index using BioBERT embeddings.
    Processes each variant's gene and condition, generates embeddings, and builds a VectorStoreIndex for retrieval.
    """
    indexed_data = []
    for _, row in variant_data.iterrows():
        text = row['GeneSymbol'] + " " + str(row['PhenotypeList'])
        embedding = generate_biobert_embeddings(text)
        indexed_data.append({"text": text, "embedding": embedding})
    
    index = VectorStoreIndex.from_documents(indexed_data)
    index.set_index_id("clinvar_index")
    return index

# Randomly select 10 variants from variant_summary
def select_random_variants(variant_data, num_variants=10):
    """
    Randomly select a specified number of variants from the variant summary data.
    Defaults to selecting 10 variants.
    """
    return variant_data.sample(n=num_variants)

# Main summarizer class
class ClinVarSummarizer:
    """
    Main class for querying ClinVar data and generating variant summaries.
    It uses a query engine to retrieve relevant variant data, synthesizes summaries, and returns the results.
    """
    def __init__(self, query_engine, conflicting_interps, variant_summary):
        self.query_engine = query_engine
        self.conflicting_interps = conflicting_interps
        self.variant_summary = variant_summary
    
    def forward(self, question):
        """
        Handles the querying and synthesis of variant summaries based on the input question.
        Extracts the gene and condition from the query, retrieves relevant data, and generates summaries.
        """
        gene, condition = parse_question(question)
        response = self.query_engine.query(question)
        context = response.response
        synthesized_summaries = synthesize_variant_summary(self.variant_summary, self.conflicting_interps, gene, condition)
        return synthesized_summaries

# Example usage
conflicting_interps_file = 'conflicting_interpretations.txt.gz'
variant_summary_file = 'variant_summary.txt.gz'

# Load ClinVar data from .txt.gz files
conflicting_interps, variant_summary = load_clinvar_data(conflicting_interps_file, variant_summary_file)

# Randomly select 10 variants
selected_variants = select_random_variants(variant_summary)

# Create an index based on the selected variant summary data
clinvar_index = create_clinvar_index(selected_variants)

# Set up the retriever
retriever = VectorIndexRetriever(index=clinvar_index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

# Initialize summarizer
summarizer = ClinVarSummarizer(query_engine=query_engine, conflicting_interps=conflicting_interps, variant_summary=selected_variants)

# Example query
question = "Please synthesize input from conflicting_interpretations.txt.gz and variant_summaries.txt.gz for KRAS G12C in prostate cancer, including PubMed citations and relevant functional studies."

# Generate and print summaries
predictions = summarizer.forward(question)
for idx, summary in enumerate(predictions, 1):
    print(f"--- Variant {idx} Summary ---")
    print(summary)
