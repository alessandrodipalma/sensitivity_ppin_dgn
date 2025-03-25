import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from src.prediction.utils import *
import pickle
import pandas as pd

def load_biogrid():
    return pickle.load(open(get_data_path() / "prediction_data/biogrid_ppi_with_nodes_having_embeddings.pkl","rb"))


def get_graph(protein_ids, biogrid_graph_uniprot, max_hop=2, verbose=False):
    undirected_biogrid = biogrid_graph_uniprot.to_undirected()
    g = undirected_biogrid.subgraph(protein_ids)

    if len(g.nodes)<2 or nx.is_connected(g) :
        return biogrid_graph_uniprot.subgraph(protein_ids).copy()
    else:
        hop=0
        components = list(nx.connected_components(g))
        node_cc_labels={}
        components_dict = {}
        added_by = {}
        for i, c in enumerate(components):
            for n in c:
                node_cc_labels[n] = {i}
                added_by[n] = set()
            components_dict[i] = set(c)

        components_count=len(components)
        needed_nodes=set()
        stop=False
        if verbose: print(f"components_dict: {components_dict}")

        def backtrack(node):
            if verbose: print("backtracking ", node)
            ancestors = added_by[node]

            for a in added_by[node]:
                if a not in ancestors:
                    ancestors.update(backtrack(a))

            return ancestors

        while not stop and (hop<max_hop):
            hop=hop+1
            if verbose: print(f"hop: {hop}")
            for n, ccs in list(node_cc_labels.items()):
                if verbose: print(f"exploring neighbors of {n}")
                for neighbor in undirected_biogrid.neighbors(n):
                    if neighbor not in node_cc_labels:
                        node_cc_labels[neighbor] = ccs
                        added_by[neighbor] = {n}
                    else:
                        added_by[neighbor].add(n)
                        components_to_merge = node_cc_labels[neighbor].symmetric_difference(ccs)
                        if len(components_to_merge)>0:
                            # the node is already in the graph, but in another component
                            # we should merge the components
                            needed_nodes.add(neighbor)
                            
                            # backtrack to add the nodes that lead to the current one
                            needed_nodes.update(backtrack(neighbor))

                            if verbose: print(f"needed nodes {needed_nodes}")
                            node_cc_labels[neighbor].update(ccs)

                            # we need to update the labels of the nodes in the components that we are merging
                            for c in node_cc_labels[neighbor]:
                                # update the nodes->components in the same ccs of neighbor
                                for other_node_in_c in components_dict[c]:
                                    node_cc_labels[other_node_in_c].update(node_cc_labels[neighbor])
                                    # if any node belongs to all the components, I can halt
                                    # but complete the whole cycle since there can be other nodes at the same hop
                                    if len(node_cc_labels[neighbor]) == components_count:
                                        stop=True

                                        if verbose: print(f"{neighbor} is in {node_cc_labels[neighbor]}. Stopping...")

                            # update the components->nodes dict
                            for c in components_to_merge:
                                for cc in components_to_merge:
                                    if c!=cc:
                                        components_dict[c] = components_dict[cc] = components_dict[c].union(components_dict[cc])                                              

        all_nodes = set(g.nodes())
        all_nodes.update(needed_nodes)
        
        final_graph = biogrid_graph_uniprot.subgraph(all_nodes).copy()
        final_graph.remove_edges_from(nx.selfloop_edges(final_graph))
        return final_graph

def nx_to_pyg(g: nx.DiGraph, pairs, embeddings: dict, io_features: bool = True) -> list:
    """ Converts all the samples of df about a model to a PyG datalist"""

    data_list = []

    node_labels = [node for node in g.nodes() if not node.startswith('->')]
    # Create a mapping of node labels to indices
    node_indices = {label: i for i, label in enumerate(node_labels)}
    
    if embeddings is None:
        node_features = np.empty((len(node_labels), 0))
    else:
        node_features = [embeddings[node] for node in node_labels]
        node_features = np.array(node_features)
    x_structure = torch.tensor(node_features, dtype=torch.float)

    # Extract edge information
    edge_indices = []
    for edge in g.edges():
        source, target = edge
        edge_indices.append((node_indices[source], node_indices[target]))

    edge_indices = list(zip(*edge_indices))
    edge_index = np.array(edge_indices)
    edge_index = torch.tensor(edge_indices, dtype=torch.long)
    for p in pairs.itertuples():
        x = torch.cat((torch.tensor([0,0]).repeat(x_structure.shape[0], 1), x_structure), dim=1)
        input_species = p.input
        output_species = p.output

        # Extract node labels
        if io_features:
            x[node_indices[input_species], 0] = 1
            x[node_indices[output_species], 1] = 1
        
        data_list.append(Data(x=x.clone().detach(), 
                              edge_index=edge_index.clone().detach(), 
                              input_species=input_species, 
                              output_species=output_species, 
                              df_index=p.Index))
   
    return data_list

def get_graph_stats(g: nx.DiGraph):
    return {
        'nodes': len(g.nodes),
        'edges': len(g.edges),
        'connected_components': nx.number_weakly_connected_components(g),
        'avg_degree': np.mean([d for n, d in g.degree()]),
        'diameter': np.max([nx.diameter(g.subgraph(c)) for c in nx.strongly_connected_components(g)]),
        'avg_clustering': nx.average_clustering(g),
    }

def print_graph_report(g: nx.DiGraph):
    stats = get_graph_stats(g)
    
    dyppin_stats = pd.read_csv(get_data_path() / "prediction_data" / "biogrid_graphs_stats.csv")
    dyppin_proteins = pd.read_csv(get_data_path() / "prediction_data" / "dyppin_proteins.txt")
    dyppin_summary = dyppin_stats.describe()
    # give the user a report of the graph with respect to the dyppin dataset
    print("Graph stats:")
    
    covered_proteins = len(set(g.nodes()).intersection(set(dyppin_proteins)))
       
    if covered_proteins==0:
        print("WARNING: the graph does not contain any protein used in the training of the model. Prediction may be less accurate")
    else:
        print(f"The model has been trained over {covered_proteins} out of the {len(g.nodes())} proteins in the graph.")

        
    warning_phrase = "Predictions may be unreliable."
    if stats['nodes'] > dyppin_summary['nodes']['max']:
        print(f"WARNING: the graph has more nodes ({stats['nodes']}) than the maximum number of nodes in the DyPPIN dataset ({dyppin_summary['nodes']['max']}). {warning_phrase}")
        
    if stats['edges'] > dyppin_summary['edges']['max']:
        print(f"WARNING: the graph has more edges ({stats['edges']}) than the maximum number of edges in the DyPPIN dataset ({dyppin_summary['edges']['max']}). {warning_phrase}")
        
    if stats['avg_degree'] > dyppin_summary['avg_degree']['max']:
        print(f"WARNING: the graph has a higher average degree ({stats['avg_degree']}) than the maximum average degree in the DYPPI dataset ({dyppin_summary['avg_degree']['max']}). {warning_phrase}")

    if stats['diameter'] > dyppin_summary['diameter']['max']:
        print(f"WARNING: the graph has a higher diameter ({stats['diameter']}) than the maximum diameter in the DyPPIN dataset ({dyppin_summary['diameter']['max']}). {warning_phrase}")
        
    
    
    

def get_input_graphs(proteins_ids: list, u_in: list, u_out: list, biogrid_graph: nx.DiGraph, embeddings_dict: dict=None):
    g = get_graph(proteins_ids, biogrid_graph)
    
    print_graph_report(g)
    # generate all possible pairs of proteins
    pairs = pd.DataFrame([(i, j) for i in u_in for j in u_out], columns=['input', 'output'])

    datalist = nx_to_pyg(g, pairs, embeddings_dict, io_features=True)

    return datalist, pairs


biogrid_base_path = get_data_path() / "external_data/biogrid/"
biogrid_identifiers_file = "BIOGRID-IDENTIFIERS-4.4.221.tab.txt"
biogrid_db_file = "BIOGRID-ALL-4.4.230.tab3.txt"

def get_biogrid_db_path():
    return biogrid_base_path / biogrid_db_file

def get_biogrid_identifiers_path():
    return biogrid_base_path / biogrid_identifiers_file
    

def get_biogrid_identifiers():
    return pd.read_csv(get_data_path() / "external_data"/"biogrid"/ biogrid_identifiers_file, sep="\t")

def get_biogrid_mappable_identifiers(identifiers_list, chunksize=50000):
    """
    Returns a dataframe containing the biogrid identifiers that are mappable to the identifiers in the identifiers list
    
    """
    if chunksize is None:
        df_biogrid_identifiers = get_biogrid_identifiers()
        print("biogrid identifiers loaded")
        biogrid_uniprot_ids = df_biogrid_identifiers[df_biogrid_identifiers['IDENTIFIER_VALUE'].isin(identifiers_list)]
    else:
        # since the identifiers file does not fit memory, read it in batch of 1000 lines
        biogrid_uniprot_ids = pd.DataFrame()
        biogrid_uniprot_ids = []
        for chunk in pd.read_csv(get_data_path() / "external_data"/"biogrid"/biogrid_identifiers_file, sep="\t", chunksize=chunksize):
            biogrid_uniprot_ids.append(chunk[chunk['IDENTIFIER_VALUE'].isin(identifiers_list)])
    
    return biogrid_uniprot_ids
    