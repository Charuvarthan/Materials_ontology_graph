"""
Advanced HEA Ontology Builder
Builds comprehensive knowledge graph from literature corpus
"""

import json
import re
import networkx as nx
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass

@dataclass
class OntologyNode:
    """Node in the HEA ontology"""
    id: str
    node_type: str  # 'alloy', 'element', 'property', 'method', 'application'
    name: str
    properties: Dict[str, Any]

@dataclass
class OntologyRelation:
    """Relationship in the HEA ontology"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]

class HEAOntologyBuilder:
    """
    Advanced ontology builder that extracts knowledge from literature corpus
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes = {}
        self.relations = []
        
        # Element symbols for validation
        self.elements = {
            'Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Zn', 'Ti', 'V', 'Nb', 'Ta', 
            'Mn', 'Mg', 'Ca', 'Sc', 'Y', 'Zr', 'Hf', 'Mo', 'W', 'Re', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'La', 'Ce', 'Pr', 'Nd', 'Gd'
        }
        
        # Synthesis methods patterns
        self.synthesis_keywords = {
            'arc melting', 'ball milling', 'mechanical alloying', 'sputtering',
            'vacuum melting', 'induction melting', 'spark plasma sintering',
            'powder metallurgy', 'hot pressing', 'vacuum induction melting'
        }
        
        # Crystal structures
        self.crystal_structures = {'BCC', 'FCC', 'HCP', 'mixed', 'amorphous'}
        
        # Applications
        self.applications = {
            'hydrogen storage', 'catalysis', 'structural', 'corrosion resistance',
            'energy storage', 'electrocatalysis', 'hydrogen production'
        }
        
    def build_ontology_from_corpus(self, papers_file: str) -> Dict[str, Any]:
        """
        Build comprehensive ontology from literature corpus
        
        Args:
            papers_file: Path to papers JSON file
            
        Returns:
            Ontology dictionary with nodes, relations, and statistics
        """
        print("🏗️ Building HEA Ontology from Literature Corpus...")
        
        # Load papers
        with open(papers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        print(f"📚 Processing {len(papers)} papers...")
        
        # Extract knowledge from each paper
        for i, paper in enumerate(papers):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(papers)} papers...")
            self._extract_knowledge_from_paper(paper, i)
        
        # Build semantic relationships
        self._build_semantic_relationships()
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        print(f"✅ Ontology built: {stats['total_nodes']} nodes, {stats['total_relations']} relations")
        
        return {
            'nodes': list(self.nodes.values()),
            'relations': self.relations,
            'statistics': stats,
            'graph': self.graph
        }
    
    def _extract_knowledge_from_paper(self, paper: Dict[str, Any], paper_id: int):
        """Extract all possible knowledge from a single paper"""
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        text = f"{title} {abstract}".lower()
        
        # Extract alloy compositions
        alloy_compositions = self._extract_alloy_compositions(title, abstract)
        
        # Extract elements from text
        elements = self._extract_elements(text)
        
        # Extract synthesis methods
        synthesis_methods = self._extract_synthesis_methods(text)
        
        # Extract crystal structures
        crystal_structures = self._extract_crystal_structures(text)
        
        # Extract applications
        applications = self._extract_applications(text)
        
        # Extract properties
        properties = self._extract_properties(text)
        
        # Create paper node
        paper_node_id = f"paper_{paper_id}"
        self._add_node(paper_node_id, 'paper', f"Paper_{paper_id}", {
            'title': title,
            'journal': paper.get('journal', ''),
            'year': paper.get('year', ''),
            'doi': paper.get('doi', '')
        })
        
        # Process alloy compositions
        for alloy in alloy_compositions:
            alloy_id = f"alloy_{alloy}"
            self._add_node(alloy_id, 'alloy_system', alloy, {
                'composition': alloy,
                'element_count': len(self._decompose_alloy(alloy))
            })
            self._add_relation(paper_node_id, alloy_id, 'mentions')
            
            # Connect alloy to its elements
            alloy_elements = self._decompose_alloy(alloy)
            for element in alloy_elements:
                element_id = f"element_{element}"
                self._add_node(element_id, 'element', element, {
                    'symbol': element,
                    'group': self._get_element_group(element)
                })
                self._add_relation(alloy_id, element_id, 'contains')
        
        # Process individual elements
        for element in elements:
            element_id = f"element_{element}"
            self._add_node(element_id, 'element', element, {
                'symbol': element,
                'group': self._get_element_group(element)
            })
            self._add_relation(paper_node_id, element_id, 'discusses')
        
        # Process synthesis methods
        for method in synthesis_methods:
            method_id = f"method_{method.replace(' ', '_')}"
            self._add_node(method_id, 'synthesis_method', method, {
                'category': self._categorize_synthesis_method(method)
            })
            self._add_relation(paper_node_id, method_id, 'uses_method')
        
        # Process crystal structures
        for structure in crystal_structures:
            structure_id = f"structure_{structure}"
            self._add_node(structure_id, 'crystal_structure', structure, {
                'lattice_type': structure
            })
            self._add_relation(paper_node_id, structure_id, 'exhibits_structure')
        
        # Process applications
        for app in applications:
            app_id = f"application_{app.replace(' ', '_')}"
            self._add_node(app_id, 'application', app, {
                'domain': self._categorize_application(app)
            })
            self._add_relation(paper_node_id, app_id, 'targets_application')
        
        # Process properties
        for prop_type, value in properties.items():
            if value:
                prop_id = f"property_{prop_type}_{str(value).replace('.', '_')}"
                self._add_node(prop_id, 'property', f"{prop_type}: {value}", {
                    'property_type': prop_type,
                    'value': value
                })
                self._add_relation(paper_node_id, prop_id, 'reports_property')
    
    def _extract_alloy_compositions(self, title: str, abstract: str) -> List[str]:
        """Extract alloy compositions from text"""
        compositions = set()
        text = f"{title} {abstract}"
        
        # Pattern for standard HEA compositions
        patterns = [
            r'\b([A-Z][a-z]?(?:[0-9]*\.?[0-9]*)?(?:[A-Z][a-z]?(?:[0-9]*\.?[0-9]*)?){2,6})\b',
            r'\b([A-Z][a-z]?(?:[A-Z][a-z]?){2,6})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if self._is_valid_alloy(match):
                    compositions.add(match)
        
        return list(compositions)
    
    def _extract_elements(self, text: str) -> Set[str]:
        """Extract chemical elements from text"""
        found_elements = set()
        for element in self.elements:
            if element.lower() in text or element in text:
                found_elements.add(element)
        return found_elements
    
    def _extract_synthesis_methods(self, text: str) -> Set[str]:
        """Extract synthesis methods from text"""
        found_methods = set()
        for method in self.synthesis_keywords:
            if method in text:
                found_methods.add(method)
        return found_methods
    
    def _extract_crystal_structures(self, text: str) -> Set[str]:
        """Extract crystal structures from text"""
        found_structures = set()
        for structure in self.crystal_structures:
            if structure.lower() in text:
                found_structures.add(structure)
        return found_structures
    
    def _extract_applications(self, text: str) -> Set[str]:
        """Extract applications from text"""
        found_apps = set()
        for app in self.applications:
            if app in text:
                found_apps.add(app)
        return found_apps
    
    def _extract_properties(self, text: str) -> Dict[str, Any]:
        """Extract numerical properties from text"""
        properties = {}
        
        # Hydrogen capacity patterns
        capacity_patterns = [
            (r'(\d+\.?\d*)\s*wt\s*%', 'hydrogen_capacity_wt'),
            (r'(\d+\.?\d*)\s*at\s*%', 'hydrogen_capacity_at'),
        ]
        
        for pattern, prop_type in capacity_patterns:
            matches = re.findall(pattern, text)
            if matches:
                properties[prop_type] = float(matches[0])
        
        # Temperature patterns
        temp_patterns = [
            (r'(\d+)\s*°c', 'temperature'),
            (r'(\d+)\s*celsius', 'temperature'),
        ]
        
        for pattern, prop_type in temp_patterns:
            matches = re.findall(pattern, text)
            if matches:
                properties[prop_type] = int(matches[0])
        
        return properties
    
    def _is_valid_alloy(self, composition: str) -> bool:
        """Check if composition is a valid HEA"""
        elements = self._decompose_alloy(composition)
        return len(elements) >= 3 and all(elem in self.elements for elem in elements)
    
    def _decompose_alloy(self, composition: str) -> List[str]:
        """Decompose alloy composition into individual elements"""
        # Remove numbers and find elements
        clean_comp = re.sub(r'[0-9.]', '', composition)
        elements = []
        i = 0
        while i < len(clean_comp):
            if i + 1 < len(clean_comp) and clean_comp[i:i+2] in self.elements:
                elements.append(clean_comp[i:i+2])
                i += 2
            elif clean_comp[i] in self.elements:
                elements.append(clean_comp[i])
                i += 1
            else:
                i += 1
        return elements
    
    def _get_element_group(self, element: str) -> str:
        """Get element group/category"""
        transition_metals = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W'}
        main_group = {'Al', 'Mg', 'Ca'}
        
        if element in transition_metals:
            return 'transition_metal'
        elif element in main_group:
            return 'main_group'
        else:
            return 'other'
    
    def _categorize_synthesis_method(self, method: str) -> str:
        """Categorize synthesis method"""
        melting_methods = ['arc melting', 'induction melting', 'vacuum melting']
        mechanical_methods = ['ball milling', 'mechanical alloying']
        deposition_methods = ['sputtering']
        
        if method in melting_methods:
            return 'melting'
        elif method in mechanical_methods:
            return 'mechanical'
        elif method in deposition_methods:
            return 'deposition'
        else:
            return 'other'
    
    def _categorize_application(self, application: str) -> str:
        """Categorize application domain"""
        energy_apps = ['hydrogen storage', 'energy storage']
        catalysis_apps = ['catalysis', 'electrocatalysis', 'hydrogen production']
        
        if application in energy_apps:
            return 'energy'
        elif application in catalysis_apps:
            return 'catalysis'
        else:
            return 'other'
    
    def _add_node(self, node_id: str, node_type: str, name: str, properties: Dict[str, Any]):
        """Add node to ontology"""
        if node_id not in self.nodes:
            node = OntologyNode(node_id, node_type, name, properties)
            self.nodes[node_id] = node
            
            # Create node attributes avoiding 'type' keyword conflict
            node_attrs = {
                'node_type': node_type,
                'name': name
            }
            # Filter out any 'type' key from properties to avoid conflict
            filtered_props = {k: v for k, v in properties.items() if k != 'type'}
            node_attrs.update(filtered_props)
            
            self.graph.add_node(node_id, **node_attrs)
    
    def _add_relation(self, source: str, target: str, relation_type: str, properties: Dict[str, Any] = None):
        """Add relation to ontology"""
        if properties is None:
            properties = {}
        
        relation = OntologyRelation(source, target, relation_type, properties)
        self.relations.append(relation)
        self.graph.add_edge(source, target, relation_type=relation_type, **properties)
    
    def _build_semantic_relationships(self):
        """Build higher-level semantic relationships"""
        print("🔗 Building semantic relationships...")
        
        # Connect alloys with similar compositions
        alloy_nodes = {nid: node for nid, node in self.nodes.items() if node.node_type == 'alloy_system'}
        
        for alloy1_id, alloy1 in alloy_nodes.items():
            for alloy2_id, alloy2 in alloy_nodes.items():
                if alloy1_id != alloy2_id:
                    similarity = self._calculate_alloy_similarity(alloy1.name, alloy2.name)
                    if similarity > 0.6:  # High similarity threshold
                        self._add_relation(alloy1_id, alloy2_id, 'similar_to', {'similarity': similarity})
    
    def _calculate_alloy_similarity(self, alloy1: str, alloy2: str) -> float:
        """Calculate similarity between two alloys"""
        elements1 = set(self._decompose_alloy(alloy1))
        elements2 = set(self._decompose_alloy(alloy2))
        
        if not elements1 or not elements2:
            return 0.0
        
        intersection = len(elements1.intersection(elements2))
        union = len(elements1.union(elements2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate ontology statistics"""
        node_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for node in self.nodes.values():
            node_types[node.node_type] += 1
        
        for relation in self.relations:
            relation_types[relation.relation_type] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_relations': len(self.relations),
            'node_types': dict(node_types),
            'relation_types': dict(relation_types),
            'graph_density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def query_ontology(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Query the ontology for information"""
        results = []
        
        if query_type == 'alloys_with_element':
            element = kwargs.get('element')
            for node_id, node in self.nodes.items():
                if node.node_type == 'alloy_system':
                    alloy_elements = self._decompose_alloy(node.name)
                    if element in alloy_elements:
                        results.append({
                            'alloy': node.name,
                            'elements': alloy_elements,
                            'node_id': node_id
                        })
        
        elif query_type == 'synthesis_methods':
            methods = {}
            for node_id, node in self.nodes.items():
                if node.node_type == 'synthesis_method':
                    category = node.properties.get('category', 'other')
                    if category not in methods:
                        methods[category] = []
                    methods[category].append(node.name)
            results = [{'category': cat, 'methods': meths} for cat, meths in methods.items()]
        
        elif query_type == 'most_studied_elements':
            element_counts = defaultdict(int)
            for relation in self.relations:
                if relation.relation_type == 'contains':
                    target_node = self.nodes.get(relation.target)
                    if target_node and target_node.node_type == 'element':
                        element_counts[target_node.name] += 1
            
            sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
            results = [{'element': elem, 'count': count} for elem, count in sorted_elements[:10]]
        
        elif query_type == 'application_domains':
            app_counts = defaultdict(int)
            for node_id, node in self.nodes.items():
                if node.node_type == 'application':
                    domain = node.properties.get('domain', 'other')
                    app_counts[domain] += 1
            results = [{'domain': domain, 'count': count} for domain, count in app_counts.items()]
        
        return results
    
    def generate_interactive_visualization(self) -> go.Figure:
        """Generate interactive Plotly visualization"""
        print("🎨 Generating interactive visualization...")
        
        # Prepare node data
        node_ids = list(self.nodes.keys())
        node_types = [self.nodes[nid].node_type for nid in node_ids]
        node_names = [self.nodes[nid].name for nid in node_ids]
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Extract positions
        x_nodes = [pos[nid][0] for nid in node_ids]
        y_nodes = [pos[nid][1] for nid in node_ids]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Color mapping for node types
        color_map = {
            'alloy_system': '#FF6B6B',
            'element': '#4ECDC4',
            'synthesis_method': '#96CEB4',
            'crystal_structure': '#45B7D1',
            'application': '#DDA0DD',
            'property': '#FFEAA7',
            'paper': '#C0C0C0'
        }
        
        node_colors = [color_map.get(nt, '#888888') for nt in node_types]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            marker=dict(size=10, color=node_colors, line=dict(width=1, color='white')),
            text=node_names,
            textposition="middle center",
            hoverinfo='text',
            hovertext=[f"{name}<br>Type: {ntype}" for name, ntype in zip(node_names, node_types)],
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title="HEA Knowledge Ontology - Interactive Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="#888", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def export_ontology(self, output_file: str):
        """Export ontology to JSON"""
        ontology_data = {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.node_type,
                    'name': node.name,
                    'properties': node.properties
                }
                for node in self.nodes.values()
            ],
            'relations': [
                {
                    'source': rel.source,
                    'target': rel.target,
                    'type': rel.relation_type,
                    'properties': rel.properties
                }
                for rel in self.relations
            ],
            'statistics': self._calculate_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Ontology exported to {output_file}")
        return ontology_data

def main():
    """Build ontology from corpus"""
    print("🚀 Starting HEA Ontology Builder")
    
    builder = HEAOntologyBuilder()
    
    # Build from existing corpus
    papers_file = "../corpus/hea_hydrogen_papers.json"
    if Path(papers_file).exists():
        ontology_data = builder.build_ontology_from_corpus(papers_file)
        builder.export_ontology("hea_ontology.json")
        print("✅ Ontology building completed!")
    else:
        print(f"❌ Papers file not found: {papers_file}")

if __name__ == "__main__":
    main()

@dataclass
class OntologyRelation:
    """Relationship in the HEA ontology"""
    source: str
    target: str
    relation_type: str  # 'contains', 'hasProperty', 'usedFor', 'comparedWith'
    properties: Dict[str, Any]

class HEAOntologyBuilder:
    """
    Builds a graph-based ontology from HEA knowledge
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes = {}
        self.relations = []
        
        # Define ontology structure
        self.node_types = {
            'alloy_system': {'color': '#FF6B6B', 'shape': 'o', 'size': 300},
            'element': {'color': '#4ECDC4', 'shape': 's', 'size': 200},
            'crystal_structure': {'color': '#45B7D1', 'shape': '^', 'size': 200},
            'synthesis_method': {'color': '#96CEB4', 'shape': 'D', 'size': 200},
            'property': {'color': '#FFEAA7', 'shape': 'v', 'size': 150},
            'application': {'color': '#DDA0DD', 'shape': 'p', 'size': 200},
            'paper': {'color': '#C0C0C0', 'shape': '.', 'size': 50},
        }
        
        self.relation_types = {
            'contains': {'color': 'blue', 'style': '-'},
            'has_structure': {'color': 'green', 'style': '--'},
            'synthesized_by': {'color': 'orange', 'style': '-.'},
            'has_property': {'color': 'red', 'style': ':'},
            'compared_with': {'color': 'purple', 'style': '-'},
            'used_for': {'color': 'brown', 'style': '-'},
            'reported_in': {'color': 'gray', 'style': '-'},
        }
    
    def build_ontology_from_knowledge(self, knowledge_file: str) -> nx.MultiDiGraph:
        """
        Build ontology graph from extracted knowledge
        
        Args:
            knowledge_file: Path to JSON file with extracted knowledge
            
        Returns:
            NetworkX graph representing the ontology
        """
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        knowledge_entries = data['knowledge']
        print(f"🏗️ Building ontology from {len(knowledge_entries)} knowledge entries...")
        
        # Process each knowledge entry
        for i, entry in enumerate(knowledge_entries):
            self._process_knowledge_entry(entry, i)
        
        print(f"✅ Ontology built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def _process_knowledge_entry(self, entry: Dict[str, Any], entry_id: int):
        """Process a single knowledge entry to add nodes and relationships"""
        
        # Create paper node
        paper_id = f"paper_{entry_id}"
        self._add_node(paper_id, 'paper', entry['title'], {
            'title': entry['title'],
            'paper_id': entry.get('paper_id', ''),
        })
        
        # Process alloy system
        if entry.get('alloy_system'):
            alloy_id = f"alloy_{entry['alloy_system']}"
            self._add_node(alloy_id, 'alloy_system', entry['alloy_system'], {
                'composition': entry['alloy_system'],
            })
            self._add_relation(paper_id, alloy_id, 'reported_in')
            
            # Process elements
            if entry.get('elements'):
                for element in entry['elements']:
                    element_id = f"element_{element}"
                    self._add_node(element_id, 'element', element, {
                        'symbol': element,
                    })
                    self._add_relation(alloy_id, element_id, 'contains')
            
            # Process crystal structure
            if entry.get('crystal_structure'):
                structure_id = f"structure_{entry['crystal_structure']}"
                self._add_node(structure_id, 'crystal_structure', entry['crystal_structure'], {
                    'structure_type': entry['crystal_structure'],  # Changed from 'type'
                })
                self._add_relation(alloy_id, structure_id, 'has_structure')
            
            # Process synthesis method
            if entry.get('synthesis_method'):
                method_id = f"method_{entry['synthesis_method'].replace(' ', '_')}"
                self._add_node(method_id, 'synthesis_method', entry['synthesis_method'], {
                    'method': entry['synthesis_method'],
                })
                self._add_relation(alloy_id, method_id, 'synthesized_by')
            
            # Process hydrogen storage properties
            if entry.get('hydrogen_capacity'):
                prop_id = f"capacity_{entry['hydrogen_capacity']}_{entry.get('hydrogen_capacity_units', 'wt')}"
                self._add_node(prop_id, 'property', f"H2 Capacity: {entry['hydrogen_capacity']} {entry.get('hydrogen_capacity_units', '')}", {
                    'property_type': 'hydrogen_capacity',  # Changed from 'type'
                    'value': entry['hydrogen_capacity'],
                    'units': entry.get('hydrogen_capacity_units', ''),
                })
                self._add_relation(alloy_id, prop_id, 'has_property')
            
            # Process temperature properties
            if entry.get('absorption_temperature'):
                temp_id = f"abs_temp_{entry['absorption_temperature']}"
                self._add_node(temp_id, 'property', f"Absorption T: {entry['absorption_temperature']}°C", {
                    'property_type': 'absorption_temperature',  # Changed from 'type'
                    'value': entry['absorption_temperature'],
                    'units': '°C',
                })
                self._add_relation(alloy_id, temp_id, 'has_property')
            
            if entry.get('desorption_temperature'):
                temp_id = f"des_temp_{entry['desorption_temperature']}"
                self._add_node(temp_id, 'property', f"Desorption T: {entry['desorption_temperature']}°C", {
                    'property_type': 'desorption_temperature',  # Changed from 'type'
                    'value': entry['desorption_temperature'],
                    'units': '°C',
                })
                self._add_relation(alloy_id, temp_id, 'has_property')
            
            # Process compared materials
            if entry.get('compared_materials'):
                for compared in entry['compared_materials']:
                    compared_id = f"alloy_{compared.replace(' ', '_')}"
                    self._add_node(compared_id, 'alloy_system', compared, {
                        'composition': compared,
                    })
                    self._add_relation(alloy_id, compared_id, 'compared_with')
        
        # Add hydrogen storage application
        storage_app_id = "app_hydrogen_storage"
        self._add_node(storage_app_id, 'application', "Hydrogen Storage", {
            'application_type': 'energy_storage',  # Changed from 'type'
            'application': 'hydrogen_storage',
        })
        if entry.get('alloy_system'):
            self._add_relation(f"alloy_{entry['alloy_system']}", storage_app_id, 'used_for')
    
    def _add_node(self, node_id: str, node_type: str, name: str, properties: Dict[str, Any]):
        """Add a node to the ontology"""
        if node_id not in self.nodes:
            node = OntologyNode(node_id, node_type, name, properties)
            self.nodes[node_id] = node
            
            # Add to NetworkX graph with all attributes
            node_attrs = {
                'node_type': node_type,  # Changed from 'type' to avoid conflict
                'name': name
            }
            node_attrs.update(properties)
            self.graph.add_node(node_id, **node_attrs)
    
    def _add_relation(self, source: str, target: str, relation_type: str, properties: Dict[str, Any] = None):
        """Add a relationship to the ontology"""
        if properties is None:
            properties = {}
        
        relation = OntologyRelation(source, target, relation_type, properties)
        self.relations.append(relation)
        
        # Add to NetworkX graph
        self.graph.add_edge(source, target, 
                           relation=relation_type, 
                           **properties)
    
    def visualize_ontology(self, output_file: str = "hea_ontology.png", layout_type: str = "spring"):
        """
        Visualize the ontology graph
        
        Args:
            output_file: Output file for the visualization
            layout_type: Layout algorithm ("spring", "circular", "hierarchical")
        """
        plt.figure(figsize=(20, 16))
        
        # Choose layout
        if layout_type == "spring":
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout_type == "hierarchical":
            pos = nx.hierarchical_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Draw nodes by type
        for node_type, style in self.node_types.items():
            nodes_of_type = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == node_type]  # Updated to use 'node_type'
            if nodes_of_type:
                nx.draw_networkx_nodes(self.graph, pos, 
                                     nodelist=nodes_of_type,
                                     node_color=style['color'],
                                     node_shape=style['shape'],
                                     node_size=style['size'],
                                     alpha=0.8)
        
        # Draw edges by type
        for relation_type, style in self.relation_types.items():
            edges_of_type = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('relation') == relation_type]
            if edges_of_type:
                nx.draw_networkx_edges(self.graph, pos,
                                     edgelist=edges_of_type,
                                     edge_color=style['color'],
                                     style=style['style'],
                                     alpha=0.6,
                                     arrows=True,
                                     arrowsize=10)
        
        # Draw labels for important nodes
        important_nodes = {n: d['name'] for n, d in self.graph.nodes(data=True) 
                          if d.get('node_type') in ['alloy_system', 'element', 'application']}  # Updated to use 'node_type'
        nx.draw_networkx_labels(self.graph, pos, important_nodes, font_size=8)
        
        # Create legend
        legend_elements = []
        for node_type, style in self.node_types.items():
            legend_elements.append(mpatches.Patch(color=style['color'], label=node_type.replace('_', ' ').title()))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("High-Entropy Alloys for Hydrogen Storage - Knowledge Ontology", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎨 Ontology visualization saved to {output_file}")
    
    def export_ontology(self, output_file: str):
        """Export ontology to JSON format"""
        ontology_data = {
            'nodes': [
                {
                    'id': node_id,
                    'type': node.node_type,  # Fixed: use node_type instead of type
                    'name': node.name,
                    'properties': node.properties
                }
                for node_id, node in self.nodes.items()
            ],
            'relations': [
                {
                    'source': rel.source,
                    'target': rel.target,
                    'type': rel.relation_type,
                    'properties': rel.properties
                }
                for rel in self.relations
            ],
            'statistics': {
                'total_nodes': len(self.nodes),
                'total_relations': len(self.relations),
                'node_types': {node_type: len([n for n in self.nodes.values() if n.node_type == node_type]) 
                              for node_type in self.node_types.keys()},
                'relation_types': {rel_type: len([r for r in self.relations if r.relation_type == rel_type]) 
                                  for rel_type in self.relation_types.keys()}
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Ontology exported to {output_file}")
        
        # Print statistics
        print("\n📊 ONTOLOGY STATISTICS:")
        print(f"Total nodes: {ontology_data['statistics']['total_nodes']}")
        print(f"Total relations: {ontology_data['statistics']['total_relations']}")
        print("Node types:", ontology_data['statistics']['node_types'])
        print("Relation types:", ontology_data['statistics']['relation_types'])
    
    def query_ontology(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Query the ontology for specific information
        
        Args:
            query_type: Type of query ('alloys_with_element', 'properties_of_alloy', etc.)
            **kwargs: Query parameters
            
        Returns:
            List of query results
        """
        results = []
        
        if query_type == 'alloys_with_element':
            element = kwargs.get('element')
            for node_id, node in self.nodes.items():
                if node.node_type == 'alloy_system':
                    # Check if alloy contains the element
                    successors = list(self.graph.successors(node_id))
                    for successor in successors:
                        successor_node = self.nodes.get(successor)
                        if (successor_node and 
                            successor_node.node_type == 'element' and 
                            successor_node.name == element):
                            results.append({
                                'alloy': node.name,
                                'id': node_id,
                                'properties': node.properties
                            })
                            break
        
        elif query_type == 'properties_of_alloy':
            alloy = kwargs.get('alloy')
            for node_id, node in self.nodes.items():
                if node.node_type == 'alloy_system' and alloy.lower() in node.name.lower():
                    # Get all properties
                    successors = list(self.graph.successors(node_id))
                    properties = []
                    for successor in successors:
                        successor_node = self.nodes.get(successor)
                        if successor_node and successor_node.node_type == 'property':
                            properties.append(successor_node.name)
                    
                    results.append({
                        'alloy': node.name,
                        'properties': properties
                    })
        
        elif query_type == 'best_capacity_alloys':
            # Find alloys with highest hydrogen capacity
            alloy_capacities = []
            for relation in self.relations:
                if relation.relation_type == 'has_property':
                    prop_node = self.nodes.get(relation.target)
                    if (prop_node and 
                        prop_node.properties.get('property_type') == 'hydrogen_capacity'):  # Updated property name
                        alloy_node = self.nodes.get(relation.source)
                        if alloy_node:
                            alloy_capacities.append({
                                'alloy': alloy_node.name,
                                'capacity': prop_node.properties.get('value'),
                                'units': prop_node.properties.get('units')
                            })
            
            # Sort by capacity
            alloy_capacities.sort(key=lambda x: x['capacity'] or 0, reverse=True)
            results = alloy_capacities[:10]  # Top 10
        
        return results

def main():
    """Main function to demonstrate ontology building"""
    print("🏗️ Starting HEA Ontology Builder")
    
    # Check if knowledge file exists
    knowledge_file = '../corpus/hea_knowledge.json'  # Adjust path as needed
    if not Path(knowledge_file).exists():
        print(f"❌ Knowledge file {knowledge_file} not found. Run hea_knowledge_extractor.py first.")
        return
    
    # Build ontology
    builder = HEAOntologyBuilder()
    ontology = builder.build_ontology_from_knowledge(knowledge_file)
    
    # Export ontology
    builder.export_ontology('hea_ontology.json')
    
    # Visualize ontology
    builder.visualize_ontology('hea_ontology.png')
    
    # Example queries
    print("\n🔍 EXAMPLE QUERIES:")
    
    # Query for alloys containing Iron
    fe_alloys = builder.query_ontology('alloys_with_element', element='Fe')
    print(f"Alloys containing Fe: {len(fe_alloys)}")
    for alloy in fe_alloys[:5]:
        print(f"  - {alloy['alloy']}")
    
    # Query for best capacity alloys
    best_alloys = builder.query_ontology('best_capacity_alloys')
    print(f"\nTop hydrogen storage alloys:")
    for alloy in best_alloys[:5]:
        print(f"  - {alloy['alloy']}: {alloy['capacity']} {alloy['units']}")

if __name__ == "__main__":
    main()
