import networkx as nx
import pickle

# Load ICD-10 ontology (assume it's a pre-built NetworkX graph)
with open('models/icd10_ontology.pkl', 'rb') as f:
    icd10_graph = pickle.load(f)

def hierarchical_search(codes: list) -> list:
    """Perform hierarchical search for ICD-10 codes."""
    expanded_codes = set(codes)
    
    for code in codes:
        # Get parent codes
        parents = list(icd10_graph.predecessors(code))
        expanded_codes.update(parents)
        
        # Get child codes
        children = list(icd10_graph.successors(code))
        expanded_codes.update(children)
        
        # Get sibling codes
        siblings = set()
        for parent in parents:
            siblings.update(icd10_graph.successors(parent))
        expanded_codes.update(siblings)
    
    return list(expanded_codes)
