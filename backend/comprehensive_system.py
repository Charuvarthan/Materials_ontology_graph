"""
Robust Enhanced HEA Knowledge System with Better Error Handling
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntry:
    """Structured knowledge entry"""
    paper_id: str
    title: str
    alloy_systems: List[str]
    elements: List[str]
    hydrogen_capacity: Optional[float]
    capacity_units: Optional[str]
    synthesis_methods: List[str]
    crystal_structures: List[str]
    key_findings: List[str]
    content: str
    relevance_score: float = 0.0

class RobustHEASystem:
    """
    Robust HEA Knowledge System with comprehensive processing
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "AIzaSyD_OFymvreIbpV5gEurkC47pMeq37ntuHc"
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Knowledge storage
        self.knowledge_entries = []
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        self.embeddings = None
        
        # Ontology graph
        self.graph = nx.MultiDiGraph()
        
        # Element patterns for extraction
        self.element_patterns = {
            'Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Zn', 'Ti', 'V', 'Nb', 'Ta', 
            'Mn', 'Mg', 'Ca', 'Sc', 'Y', 'Zr', 'Hf', 'Mo', 'W', 'Re', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'La', 'Ce', 'Pr', 'Nd', 'Gd'
        }
    
    def process_full_corpus_robust(self, papers_file: str) -> None:
        """Process the complete corpus with robust error handling"""
        print("🚀 Starting Robust Full Corpus Processing...")
        
        with open(papers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data['papers']
        print(f"📚 Processing {len(papers)} research papers...")
        
        successful_extractions = 0
        
        for i, paper in enumerate(papers, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(papers)} papers processed...")
            
            try:
                # Extract knowledge with fallback methods
                knowledge = self._extract_robust_knowledge(paper, i)
                if knowledge:
                    self.knowledge_entries.append(knowledge)
                    self._add_to_graph(knowledge)
                    successful_extractions += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process paper {i}: {e}")
                continue
        
        # Build embeddings
        self._build_robust_embeddings()
        
        print(f"Successfully processed {successful_extractions}/{len(papers)} papers")
    
    def _extract_robust_knowledge(self, paper: Dict[str, Any], paper_id: int) -> Optional[KnowledgeEntry]:
        """Extract knowledge with multiple fallback methods"""
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        if not title and not abstract:
            return None
        
        # Method 1: AI Extraction with simple prompt
        ai_knowledge = self._try_ai_extraction(title, abstract)
        
        # Method 2: Rule-based extraction as fallback
        rule_knowledge = self._rule_based_extraction(title, abstract)
        
        # Combine knowledge sources
        combined_knowledge = self._combine_knowledge(ai_knowledge, rule_knowledge)
        
        # Create structured entry
        content = f"Title: {title} | Abstract: {abstract}"
        if combined_knowledge['key_findings']:
            content += f" | Key Findings: {'; '.join(combined_knowledge['key_findings'])}"
        
        entry = KnowledgeEntry(
            paper_id=paper.get('doi', f"paper_{paper_id}"),
            title=title,
            alloy_systems=combined_knowledge['alloy_systems'],
            elements=combined_knowledge['elements'],
            hydrogen_capacity=combined_knowledge['hydrogen_capacity'],
            capacity_units=combined_knowledge['capacity_units'],
            synthesis_methods=combined_knowledge['synthesis_methods'],
            crystal_structures=combined_knowledge['crystal_structures'],
            key_findings=combined_knowledge['key_findings'],
            content=content
        )
        
        return entry
    
    def _try_ai_extraction(self, title: str, abstract: str) -> Dict[str, Any]:
        """Try AI extraction with robust prompting"""
        try:
            prompt = f"""
            Extract key information from this HEA paper. Respond with ONLY a simple structured format:
            
            ALLOY_SYSTEMS: [list any alloy compositions like FeCoNiCuZn, AlCrFeNi, etc.]
            ELEMENTS: [list chemical elements: Fe, Co, Ni, Al, etc.]
            HYDROGEN_CAPACITY: [number if mentioned, otherwise NONE]
            CAPACITY_UNITS: [wt% or at% if capacity mentioned, otherwise NONE]
            SYNTHESIS: [methods like arc melting, ball milling, etc.]
            CRYSTAL_STRUCTURE: [BCC, FCC, HCP if mentioned, otherwise NONE]
            KEY_FINDINGS: [2-3 most important discoveries]
            
            Paper:
            Title: {title}
            Abstract: {abstract}
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse structured response
            knowledge = {
                'alloy_systems': [],
                'elements': [],
                'hydrogen_capacity': None,
                'capacity_units': None,
                'synthesis_methods': [],
                'crystal_structures': [],
                'key_findings': []
            }
            
            # Parse each section
            sections = response_text.split('\n')
            for line in sections:
                line = line.strip()
                if line.startswith('ALLOY_SYSTEMS:'):
                    alloys = re.findall(r'[A-Z][a-z]?(?:[A-Z][a-z]?){2,6}', line)
                    knowledge['alloy_systems'] = alloys
                elif line.startswith('ELEMENTS:'):
                    elements = [e.strip() for e in re.findall(r'[A-Z][a-z]?', line) if e in self.element_patterns]
                    knowledge['elements'] = elements
                elif line.startswith('HYDROGEN_CAPACITY:'):
                    capacity_match = re.search(r'(\d+\.?\d*)', line)
                    if capacity_match:
                        knowledge['hydrogen_capacity'] = float(capacity_match.group(1))
                elif line.startswith('CAPACITY_UNITS:'):
                    if 'wt%' in line:
                        knowledge['capacity_units'] = 'wt%'
                    elif 'at%' in line:
                        knowledge['capacity_units'] = 'at%'
                elif line.startswith('SYNTHESIS:'):
                    methods = []
                    if 'arc melting' in line.lower():
                        methods.append('arc melting')
                    if 'ball milling' in line.lower():
                        methods.append('ball milling')
                    if 'mechanical alloying' in line.lower():
                        methods.append('mechanical alloying')
                    knowledge['synthesis_methods'] = methods
                elif line.startswith('CRYSTAL_STRUCTURE:'):
                    structures = []
                    if 'BCC' in line:
                        structures.append('BCC')
                    if 'FCC' in line:
                        structures.append('FCC')
                    if 'HCP' in line:
                        structures.append('HCP')
                    knowledge['crystal_structures'] = structures
                elif line.startswith('KEY_FINDINGS:'):
                    findings = [line.replace('KEY_FINDINGS:', '').strip()]
                    knowledge['key_findings'] = findings
            
            return knowledge
            
        except Exception as e:
            logger.warning(f"AI extraction failed: {e}")
            return {}
    
    def _rule_based_extraction(self, title: str, abstract: str) -> Dict[str, Any]:
        """Rule-based extraction as fallback"""
        text = f"{title} {abstract}".lower()
        
        knowledge = {
            'alloy_systems': [],
            'elements': [],
            'hydrogen_capacity': None,
            'capacity_units': None,
            'synthesis_methods': [],
            'crystal_structures': [],
            'key_findings': []
        }
        
        # Extract alloy compositions
        alloy_patterns = [
            r'\b([A-Z][a-z]?(?:[A-Z][a-z]?){2,6})\b',
            r'\b([A-Z][a-z]?(?:[0-9.]+[A-Z][a-z]?){2,6})\b'
        ]
        
        original_text = f"{title} {abstract}"
        for pattern in alloy_patterns:
            matches = re.findall(pattern, original_text)
            for match in matches:
                if self._is_valid_alloy(match):
                    knowledge['alloy_systems'].append(match)
        
        # Extract elements
        for element in self.element_patterns:
            if element.lower() in text or element in original_text:
                knowledge['elements'].append(element)
        
        # Extract hydrogen capacity
        capacity_patterns = [
            r'(\d+\.?\d*)\s*wt\s*%',
            r'(\d+\.?\d*)\s*at\s*%',
            r'(\d+\.?\d*)\s*weight\s*%'
        ]
        
        for pattern in capacity_patterns:
            match = re.search(pattern, text)
            if match:
                knowledge['hydrogen_capacity'] = float(match.group(1))
                if 'wt' in pattern:
                    knowledge['capacity_units'] = 'wt%'
                elif 'at' in pattern:
                    knowledge['capacity_units'] = 'at%'
                break
        
        # Extract synthesis methods
        synthesis_keywords = ['arc melting', 'ball milling', 'mechanical alloying', 'sputtering']
        for method in synthesis_keywords:
            if method in text:
                knowledge['synthesis_methods'].append(method)
        
        # Extract crystal structures
        if 'bcc' in text:
            knowledge['crystal_structures'].append('BCC')
        if 'fcc' in text:
            knowledge['crystal_structures'].append('FCC')
        if 'hcp' in text:
            knowledge['crystal_structures'].append('HCP')
        
        return knowledge
    
    def _is_valid_alloy(self, composition: str) -> bool:
        """Check if composition is a valid HEA"""
        elements = re.findall(r'[A-Z][a-z]?', composition)
        return len(elements) >= 3 and all(elem in self.element_patterns for elem in elements)
    
    def _combine_knowledge(self, ai_knowledge: Dict[str, Any], rule_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Combine AI and rule-based knowledge"""
        combined = {
            'alloy_systems': list(set((ai_knowledge.get('alloy_systems', []) + rule_knowledge.get('alloy_systems', [])))),
            'elements': list(set((ai_knowledge.get('elements', []) + rule_knowledge.get('elements', [])))),
            'hydrogen_capacity': ai_knowledge.get('hydrogen_capacity') or rule_knowledge.get('hydrogen_capacity'),
            'capacity_units': ai_knowledge.get('capacity_units') or rule_knowledge.get('capacity_units'),
            'synthesis_methods': list(set((ai_knowledge.get('synthesis_methods', []) + rule_knowledge.get('synthesis_methods', [])))),
            'crystal_structures': list(set((ai_knowledge.get('crystal_structures', []) + rule_knowledge.get('crystal_structures', [])))),
            'key_findings': ai_knowledge.get('key_findings', []) + rule_knowledge.get('key_findings', [])
        }
        return combined
    
    def _add_to_graph(self, knowledge: KnowledgeEntry) -> None:
        """Add knowledge to the ontology graph"""
        # Add paper node
        self.graph.add_node(knowledge.paper_id, 
                           type='paper',
                           title=knowledge.title)
        
        # Add alloy systems
        for alloy in knowledge.alloy_systems:
            alloy_id = f"alloy_{alloy}"
            self.graph.add_node(alloy_id, type='alloy', composition=alloy)
            self.graph.add_edge(knowledge.paper_id, alloy_id, relation='studies')
        
        # Add elements
        for element in knowledge.elements:
            elem_id = f"element_{element}"
            self.graph.add_node(elem_id, type='element', symbol=element)
            self.graph.add_edge(knowledge.paper_id, elem_id, relation='discusses')
        
        # Add properties
        if knowledge.hydrogen_capacity:
            prop_id = f"h2_cap_{knowledge.hydrogen_capacity}"
            self.graph.add_node(prop_id, 
                              type='property',
                              property_type='hydrogen_capacity',
                              value=knowledge.hydrogen_capacity,
                              units=knowledge.capacity_units)
            self.graph.add_edge(knowledge.paper_id, prop_id, relation='reports')
    
    def _build_robust_embeddings(self) -> None:
        """Build embeddings for knowledge retrieval"""
        if not self.knowledge_entries:
            return
        
        contents = [entry.content for entry in self.knowledge_entries]
        self.embeddings = self.vectorizer.fit_transform(contents)
        print(f"🔍 Built embeddings for {len(contents)} knowledge entries")
    
    def enhanced_search(self, query: str, top_k: int = 5) -> List[KnowledgeEntry]:
        """Enhanced semantic search"""
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []
        
        # Transform query
        query_embedding = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Minimum relevance
                entry = self.knowledge_entries[idx]
                entry.relevance_score = similarities[idx]
                results.append(entry)
        
        return results
    
    def intelligent_qa(self, question: str) -> Dict[str, Any]:
        """Intelligent Q&A with comprehensive knowledge retrieval"""
        print(f"Processing: {question}")
        
        # Search for relevant knowledge
        relevant_entries = self.enhanced_search(question, top_k=10)
        
        if not relevant_entries:
            return {
                'answer': "I don't have sufficient relevant information to answer this question.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Build comprehensive context
        context_parts = []
        sources = []
        
        for entry in relevant_entries[:5]:
            context_parts.append(f"Study: {entry.title}")
            
            if entry.alloy_systems:
                context_parts.append(f"Alloys: {', '.join(entry.alloy_systems)}")
            
            if entry.hydrogen_capacity:
                context_parts.append(f"H2 Capacity: {entry.hydrogen_capacity} {entry.capacity_units or ''}")
            
            if entry.synthesis_methods:
                context_parts.append(f"Synthesis: {', '.join(entry.synthesis_methods)}")
            
            if entry.key_findings:
                context_parts.append(f"Findings: {'; '.join(entry.key_findings[:2])}")
            
            context_parts.append("---")
            
            sources.append({
                'title': entry.title,
                'alloy_systems': entry.alloy_systems,
                'relevance': float(entry.relevance_score)
            })
        
        context = "\\n".join(context_parts)
        
        # Generate comprehensive answer
        try:
            prompt = f"""
            You are an expert in High-Entropy Alloys for hydrogen storage. Answer the question comprehensively based on the research context provided.
            
            Context from literature:
            {context}
            
            Question: {question}
            
            Provide a detailed, scientifically accurate answer that includes:
            1. Direct answer to the question
            2. Specific alloy compositions and their properties
            3. Supporting evidence from the studies
            4. Practical implications
            5. Current research trends
            
            Answer:
            """
            
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Calculate confidence
            avg_relevance = np.mean([entry.relevance_score for entry in relevant_entries[:5]])
            confidence = min(avg_relevance * 3, 1.0)
            
            return {
                'answer': answer,
                'confidence': float(confidence),
                'sources': sources,
                'num_relevant_studies': len(relevant_entries)
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {e}",
                'confidence': 0.0,
                'sources': sources
            }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'total_papers': len(self.knowledge_entries),
            'graph_nodes': len(self.graph.nodes),
            'graph_edges': len(self.graph.edges)
        }
        
        # Analyze knowledge coverage
        all_alloys = set()
        all_elements = set()
        capacities = []
        
        for entry in self.knowledge_entries:
            all_alloys.update(entry.alloy_systems)
            all_elements.update(entry.elements)
            if entry.hydrogen_capacity:
                capacities.append(entry.hydrogen_capacity)
        
        stats.update({
            'unique_alloy_systems': len(all_alloys),
            'unique_elements': len(all_elements),
            'papers_with_capacity_data': len(capacities),
            'avg_capacity': np.mean(capacities) if capacities else 0,
            'max_capacity': max(capacities) if capacities else 0,
            'top_elements': list(all_elements)[:15],
            'sample_alloys': list(all_alloys)[:10]
        })
        
        return stats
    
    def save_comprehensive_system(self, output_dir: str) -> None:
        """Save the complete system"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save knowledge entries
        entries_data = [asdict(entry) for entry in self.knowledge_entries]
        
        system_data = {
            'system_stats': self.get_comprehensive_stats(),
            'knowledge_entries': entries_data,
            'total_processed': len(self.knowledge_entries)
        }
        
        with open(f"{output_dir}/comprehensive_knowledge.json", 'w', encoding='utf-8') as f:
            json.dump(system_data, f, indent=2, ensure_ascii=False)
        
        # Save graph
        nx.write_gml(self.graph, f"{output_dir}/comprehensive_ontology.gml")
        
        print(f"Comprehensive system saved to {output_dir}")

def main():
    """Run the comprehensive enhanced system"""
    print("Starting Comprehensive HEA Knowledge System")
    
    # Initialize system
    system = RobustHEASystem()
    
    # Process full corpus
    papers_file = 'corpus/hea_hydrogen_papers.json'
    if Path(papers_file).exists():
        system.process_full_corpus_robust(papers_file)
        
        # Show comprehensive stats
        stats = system.get_comprehensive_stats()
        print("\nCOMPREHENSIVE SYSTEM STATISTICS:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Demonstrate enhanced Q&A
        advanced_questions = [
            "What are the highest performing HEA compositions for hydrogen storage and what capacities do they achieve?",
            "How do different synthesis methods impact the hydrogen storage properties of HEAs?",
            "What crystal structures are most favorable for hydrogen storage in HEAs?",
            "Which specific alloy systems show the best combination of capacity and stability?"
        ]
        
        print("\nADVANCED Q&A DEMONSTRATIONS:")
        for i, question in enumerate(advanced_questions[:2], 1):
            print(f"\n{'='*50}")
            print(f"Question {i}: {question}")
            result = system.intelligent_qa(question)
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {result['num_relevant_studies']}")
            print(f"Answer: {result['answer']}")
            print('='*50)
        
        # Save comprehensive system
        system.save_comprehensive_system('comprehensive_system')
        
        print("\\nComprehensive HEA Knowledge System is ready for advanced queries!")
        
    else:
        print(f"Papers file not found: {papers_file}")

if __name__ == "__main__":
    main()
