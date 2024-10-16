import networkx as nx

# Create a simple mock ICD-10 ontology
icd10_graph = nx.DiGraph()
icd10_graph.add_edges_from([
    ('I', 'I21'), ('I21', 'I21.3'),
    ('I', 'I25'), ('I25', 'I25.10'),
    ('I', 'I10'),
    ('J', 'J15'), ('J15', 'J15.9'),
    ('J', 'J96'), ('J96', 'J96.01'),
    ('E', 'E11'), ('E11', 'E11.9'),
    ('E', 'E78'), ('E78', 'E78.5')
])

def hierarchical_search(codes: list) -> list:
    """Perform hierarchical search for ICD-10 codes."""
    expanded_codes = set(codes)
    
    for code in codes:
        # Get parent codes
        parents = list(nx.ancestors(icd10_graph, code))
        expanded_codes.update(parents)
        
        # Get child codes
        children = list(nx.descendants(icd10_graph, code))
        expanded_codes.update(children)
        
        # Get sibling codes
        siblings = set()
        for parent in parents:
            siblings.update(icd10_graph.successors(parent))
        expanded_codes.update(siblings)
    
    return list(expanded_codes)
