import requests
import pandas as pd
import os
import time
from pybiomart import Server
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_ensembl_to_reactome():
    """
    Downloads the Ensembl2Reactome.txt file from Reactome,
    filters for human entries, and creates a pathway dictionary.
    """
    # URL for the Ensembl2Reactome.txt file
    url = "https://reactome.org/download/current/Ensembl2Reactome.txt"
    
    print(f"Downloading Ensembl2Reactome.txt from {url}...")
    
    try:
        # Download the file
        # response = requests.get(url, timeout=60)
        # response.raise_for_status()  # Raise exception for HTTP errors
        
        # # Save the raw file
        # with open("Ensembl2Reactome.txt", "wb") as f:
        #     f.write(response.content)
        
        print("Download complete. Processing file...")
        
        # Read the file into a pandas DataFrame - adjust columns based on your sample
        df = pd.read_csv("Ensembl2Reactome.txt", sep="\t", header=None)
        
        # Based on your sample, set column names appropriately
        # The exact column names may need adjustment based on the actual data
        df.columns = ["ensembl_id", "reactome_id", "url", "pathway_name", "evidence", "species"]
        
        # Print the first few rows to verify
        print("First few rows of the data:")
        print(df.head())
        
        # Filter for human entries
        human_df = df[df["species"] == "Homo sapiens"]
        
        # Save human-specific mapping
        human_df.to_csv("Ensembl2Reactome_Human.txt", sep="\t", index=False)
        
        print(f"Filtered human entries: {len(human_df)} rows saved to Ensembl2Reactome_Human.txt")
        print(human_df.head())
        
        # Create pathway dictionary for MORE framework
        # Format: {pathway_id: [ensembl_id1, ensembl_id2, ...]}
        pathway_dict = {}
        for pathway_id, group in human_df.groupby("reactome_id"):
            genes = group["ensembl_id"].unique().tolist()
            pathway_dict[pathway_id] = genes
        
        print(f"Created pathway dictionary with {len(pathway_dict)} pathways")

        # Basic stats about the pathways
        pathway_sizes = [len(genes) for genes in pathway_dict.values()]
        avg_genes_per_pathway = sum(pathway_sizes) / len(pathway_sizes)
        
        print(f"Average genes per pathway: {avg_genes_per_pathway:.2f}")
        print(f"Smallest pathway: {min(pathway_sizes)} genes")
        print(f"Largest pathway: {max(pathway_sizes)} genes")
        
        # Return the pathway dictionary
        return pathway_dict
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        # Print the first few lines of the file to help debug the format
        try:
            with open("Ensembl2Reactome.txt", "r") as f:
                print("First 5 lines of the file:")
                for i, line in enumerate(f):
                    if i < 5:
                        print(line.strip())
                    else:
                        break
        except Exception:
            pass
        return None

def filter_pathways_by_gene_list(pathway_dict, gene_list):
    """
    Filters the pathway dictionary to only include pathways
    that contain at least one gene from the provided gene list.
    
    Args:
        pathway_dict: Dictionary mapping pathway IDs to gene lists
        gene_list: List of ENSEMBL gene IDs you're interested in
    
    Returns:
        Filtered pathway dictionary
    """
    print("Gene list")
    print(gene_list[1:10])
    gene_set = set(gene_list)
    filtered_dict = {}
    
    for pathway_id, genes in pathway_dict.items():
        # Check for overlap between pathway genes and your gene list
        common_genes = set(genes).intersection(gene_set)
        
        if common_genes:
            filtered_dict[pathway_id] = genes
    
    print(f"Filtered from {len(pathway_dict)} to {len(filtered_dict)} pathways " 
          f"that contain genes from your list")
    
    return filtered_dict

def convert_gene_symbols_to_ensembl(gene_symbols):
    """Convert gene symbols to ENSEMBL IDs using BioMart."""
    logger.info(f"Converting {len(gene_symbols)} gene symbols to ENSEMBL IDs...")
    
    # Connect to Ensembl BioMart
    server = Server(host="http://www.ensembl.org")
    mart = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    
    # Get mapping - handle large lists by processing in batches
    symbol_to_ensembl = {}
    
    # for i in range(0, len(gene_symbols), batch_size):
    try:
        results = mart.query(attributes=["external_gene_name", "ensembl_gene_id"])
        
        # Add to our dictionary
        batch_dict = dict(zip(results["Gene name"], results["Gene stable ID"]))
        symbol_to_ensembl.update(batch_dict)
        
        # Be nice to the server
        time.sleep(0.5)
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
    
    # Check for missing conversions
    missing = set(gene_symbols) - set(symbol_to_ensembl.keys())
    if missing:
        logger.warning(f"Could not find ENSEMBL IDs for {len(missing)} gene symbols")
        logger.debug(f"Missing symbols: {list(missing)[:10]}...")

    filtered_symbol_to_ensembl = {}

    for symbol in gene_symbols:
        if symbol in symbol_to_ensembl.keys():
            filtered_symbol_to_ensembl[symbol] = symbol_to_ensembl[symbol]
    
    logger.info(f"Successfully converted {len(filtered_symbol_to_ensembl)} genes")
    return filtered_symbol_to_ensembl

# Example usage
if __name__ == "__main__":
    # Download and process the full Ensembl2Reactome file
    pathway_dict = download_ensembl_to_reactome()
    
    # If successful, save the pathway dictionary
    if pathway_dict:
        import json
        with open("reactome_pathways.json", "w") as f:
            json.dump(pathway_dict, f)
        
        print("Pathway dictionary saved to reactome_pathways.json")
    
