# ClinVar Variant Summarization Tool

This repository provides a tool to synthesize clinical summaries of genetic variants from the **ClinVar** database. The tool automatically processes genetic variant data, identifies conflicting clinical interpretations, and extracts PubMed citations from ClinVar’s flat files. Using **BioBERT** embeddings for advanced querying, the tool can generate variant summaries for a given gene and clinical condition.

## Disclaimer and Intended Use
This project is intended for research and informational purposes only. The variant syntheses generated by this workflow are preliminary interpretations based on publicly available data (e.g., ClinVar). These summaries are meant to support clinical review but should not be considered final or definitive interpretations for clinical decision-making.
All variant syntheses must be carefully reviewed and validated by a qualified healthcare professional or clinical geneticist before being used in any diagnostic or therapeutic context. This tool does not replace professional medical advice, diagnosis, or treatment.
The authors of this project are not liable for any use of the data or interpretations generated by this workflow that results in misdiagnosis or harm. Always consult a healthcare professional for genetic counseling and interpretation of genetic results.

### No Warranty
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or contributors be held liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

By using this software, you acknowledge and agree that the authors and contributors are not liable for any outcomes resulting from the use of the data or interpretations generated by this tool. Users must ensure that all outputs are used responsibly and validated by appropriate professional standards.
Always consult a healthcare professional for genetic counseling and final interpretation of genetic data.


## Features
- **Reads and processes ClinVar `.txt.gz` flat files** for both **conflicting interpretations** and **variant summaries**.
- **Randomly selects 10 variants** from the variant summary data for processing.
- **Uses BioBERT embeddings** to represent and index variant data for efficient similarity-based querying.
- **Generates comprehensive clinical summaries** that include:
  - Clinical significance.
  - Review status.
  - Conflicting clinical interpretations.
  - Relevant PubMed citations extracted from the dataset.
- **Automates clinical variant interpretation** based on the user's query, using natural language processing.

## How It Works

1. **Input Data**: The tool reads the ClinVar flat files, specifically `conflicting_interpretations.txt.gz` and `variant_summary.txt.gz`.
2. **Variant Selection**: 10 variants are randomly selected from the variant summary file.
3. **Embedding**: BioBERT embeddings are generated for each variant, which represent the gene and phenotype data in a way that can be indexed and queried.
4. **Query Processing**: Users can input natural language queries, such as asking for the clinical impact of a variant in a specific condition.
5. **Summary Generation**: For each queried variant, the tool generates a structured summary that includes:
   - Variant details (e.g., gene, condition, clinical significance).
   - Conflicting interpretations (if any).
   - PubMed citations.
6. **Customization**: The tool can be adapted to suit specific clinical workflows or to process additional variants beyond the randomly selected set.

## Installation

To use this tool, you need to have Python 3.x installed, along with several dependencies:

1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/clinvar-variant-summarizer.git
    cd clinvar-variant-summarizer
    ```

2. Install the required dependencies:
    ```bash
    pip install pandas transformers llama-index
    ```

3. Ensure that the ClinVar `.txt.gz` flat files are in the same directory as the script:
    - `conflicting_interpretations.txt.gz`
    - `variant_summary.txt.gz`
    
   These can be downloaded from the [ClinVar FTP directory](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/).

## Usage

Once you have everything set up, you can run the script and query the variant data.

1. **Run the script**:
    ```bash
    python variant_summarizer.py
    ```

2. **Example Query**:
    The script will prompt you for a question. You can input a query like:
    ```text
    Please synthesize input from conflicting_interpretations.txt.gz and variant_summaries.txt.gz for KRAS G12C in prostate cancer, including PubMed citations and relevant functional studies.
    ```

3. **Output**:
    The tool will output a clinical summary of the variant, including information like:
    - Clinical significance.
    - Conflicting interpretations (if any).
    - Extracted PubMed citations.

    The output might look like this:
    ```text
    --- Variant 1 Summary ---
    Gene: KRAS
    Condition: prostate cancer
    Clinical Significance: Pathogenic
    Review Status: criteria provided, single submitter
    Associated Conditions: prostate cancer
    Conflicting Clinical Interpretations: Submitter1: Pathogenic, Submitter2: Likely Benign
    Relevant literature references (PMIDs): 12345678, 87654321
    This synthesis is a preliminary interpretation. Please review and modify for accuracy.
    ```

## Key Components

- **BioBERT Embeddings**: The script uses BioBERT (`dmis-lab/biobert-base-cased-v1.1`) to create embeddings for gene and phenotype data, which are then used for similarity-based querying.
- **VectorStoreIndex**: The variant data is indexed using **VectorStoreIndex** for efficient retrieval.
- **Summarization**: A structured clinical summary is generated for each variant, which can be used as a template for pathologists to modify and review.

## Future Improvements

- **Prompt Engineering**, or automated optimization of prompt refinement (e.g. automated with DSPy).
- **Extend Variant Selection**: Modify the script to allow users to select more variants or filter them based on specific criteria.
- **Enhanced Querying**: Integrate more advanced NLP techniques for better parsing of user queries.
- **Support for Additional Flat Files**: Extend support for more fields from the ClinVar dataset, such as inheritance modes or population frequencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

**Contributors**:  
- Vincent Laufer, CGO Open Chromatin Group, LLC - a topologically associated domain of chromatin cloud. Email: laufer@openchromatin.com
