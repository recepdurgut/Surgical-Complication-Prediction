# In Silico Detection of AMR Genes in Bacterial Genome

This project focuses on identifying known **Antimicrobial Resistance (AMR)** genes in an *E. coli* genome using computational tools, Python, and the **CARD (Comprehensive Antibiotic Resistance Database)**.

## 🧬 Biological Background
Antibiotic resistance is a global health crisis. This pipeline uses **BLASTX** to translate nucleotide sequences and align them against curated AMR protein models. 

### Key Mechanisms Analyzed:
- **Efflux Pumps:** (e.g., AcrB, AcrF)
- **Regulatory Systems:** (e.g., EvgS)
- **Target Modification & Transport:** (e.g., MsbA)

## 🛠️ Tools & Technologies
- **Language:** Python 3.x
- **Libraries:** Biopython, Pandas, Seaborn, Matplotlib, Numpy
- **Bioinformatics Tools:** BLAST+ (makeblastdb, blastx)
- **Database:** CARD (Protein Homolog Models)

## 📊 Methodology
1. **Database Preparation:** Converting CARD protein FASTA into a searchable BLAST database.
2. **Alignment:** Running BLASTX with the *E. coli* genome against the CARD database.
3. **Filtering:** Hits are filtered based on:
   - **Identity > 90%**
   - **E-value < 1e-5**
4. **Visualization:** Statistical distribution of identity and alignment lengths.

## 📈 Results
The pipeline successfully identified **45 high-confidence AMR gene candidates**. 
- **40 out of 45 hits** showed 100% identity, indicating highly conserved resistance mechanisms.
- Dominant findings include RND-type efflux systems, which are primary contributors to multidrug resistance.

## 📂 Project Structure
```text
├── sequence.fasta          # Target bacterial genome
├── card_proteins.fasta     # Reference AMR database (CARD)
├── amr_detection.ipynb     # Main analysis notebook
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
