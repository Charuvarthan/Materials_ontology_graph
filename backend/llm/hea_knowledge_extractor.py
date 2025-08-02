"""
Knowledge Extractor for HEA Literature
Uses LLM to extract structured knowledge from HEA papers for ontology mapping
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import openai
from pathlib import Path

# Import our custom modules
import sys
sys.path.append('..')
from llm.gemini_extractor import GeminiExtractor

@dataclass
class HEAKnowledge:
    """Structured knowledge extracted from HEA papers"""
    # Paper identification
    paper_id: str
    title: str
    
    # Material properties
    alloy_system: str  # e.g., "AlCoCrFeNi"
    elements: List[str]  # e.g., ["Al", "Co", "Cr", "Fe", "Ni"]
    crystal_structure: Optional[str]  # e.g., "BCC", "FCC", "mixed"
    
    # Hydrogen storage properties
    hydrogen_capacity: Optional[float]  # wt%
    hydrogen_capacity_units: Optional[str]  # "wt%" or "at%"
    absorption_temperature: Optional[float]  # °C
    desorption_temperature: Optional[float]  # °C
    pressure_conditions: Optional[float]  # bar
    
    # Kinetic properties
    absorption_time: Optional[float]  # minutes
    desorption_time: Optional[float]  # minutes
    cyclic_stability: Optional[int]  # number of cycles
    
    # Processing information
    synthesis_method: Optional[str]  # e.g., "arc melting", "ball milling"
    heat_treatment: Optional[str]
    
    # Performance metrics
    reversibility: Optional[str]  # "reversible", "irreversible", "partial"
    activation_energy: Optional[float]  # kJ/mol
    
    # Additional properties
    phase_composition: List[str]  # phases present
    microstructure: Optional[str]
    
    # Relationships and comparisons
    compared_materials: List[str]
    advantages: List[str]
    disadvantages: List[str]
    
    # Research context
    research_focus: str  # e.g., "fundamental study", "application development"
    key_findings: List[str]

class HEAKnowledgeExtractor:
    """
    Extract structured knowledge from HEA papers using LLM
    """
    
    def __init__(self, use_gemini=True):
        """
        Initialize the knowledge extractor
        
        Args:
            use_gemini: If True, use Gemini API; if False, use local LLM
        """
        self.use_gemini = use_gemini
        
        if use_gemini:
            # Pass the API key directly
            api_key = "AIzaSyD_OFymvreIbpV5gEurkC47pMeq37ntuHc"
            self.extractor = GeminiExtractor(api_key=api_key)
        else:
            # We'll implement local LLM later
            raise NotImplementedError("Local LLM not implemented yet")
    
    def extract_knowledge_from_papers(self, papers_file: str) -> List[HEAKnowledge]:
        """
        Extract knowledge from a JSON file of papers
        
        Args:
            papers_file: Path to JSON file with papers
            
        Returns:
            List of HEAKnowledge objects
        """
        with open(papers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data['papers']
        print(f"📚 Extracting knowledge from {len(papers)} papers...")
        
        extracted_knowledge = []
        
        # Process ALL papers for comprehensive knowledge extraction
        for i, paper in enumerate(papers, 1):
            print(f"\n🧠 Processing paper {i}/{len(papers)}: {paper['title'][:50]}...")
            
            knowledge = self.extract_knowledge_from_paper(paper)
            if knowledge:
                extracted_knowledge.append(knowledge)
                print(f"✅ Extracted knowledge for paper {i}")
            else:
                print(f"❌ Failed to extract knowledge for paper {i}")
        
        return extracted_knowledge
    
    def extract_knowledge_from_paper(self, paper: Dict[str, Any]) -> Optional[HEAKnowledge]:
        """
        Extract structured knowledge from a single paper
        
        Args:
            paper: Paper dictionary from JSON
            
        Returns:
            HEAKnowledge object or None if extraction fails
        """
        try:
            # Create extraction prompt
            prompt = self._create_extraction_prompt(paper)
            
            # Use LLM to extract knowledge
            response = self.extractor.extract_knowledge_from_paper(
                paper['title'], 
                paper.get('abstract', ''), 
                paper.get('doi', '')
            )
            
            # Parse response into HEAKnowledge object
            if hasattr(response, '__dict__'):
                # Convert ExtractedKnowledge to HEAKnowledge
                knowledge = HEAKnowledge(
                    paper_id=response.paper_id,
                    title=response.title,
                    alloy_system=response.alloy_system,
                    elements=response.elements or [],
                    crystal_structure=response.crystal_structure,
                    hydrogen_capacity=response.hydrogen_capacity,
                    hydrogen_capacity_units="wt%" if response.hydrogen_capacity else None,
                    absorption_temperature=response.temperature,
                    desorption_temperature=None,
                    pressure_conditions=response.pressure,
                    absorption_time=None,
                    desorption_time=None,
                    cyclic_stability=None,
                    synthesis_method=response.synthesis_method,
                    heat_treatment=None,
                    reversibility=None,
                    activation_energy=None,
                    phase_composition=[],
                    microstructure=None,
                    compared_materials=[],
                    advantages=[],
                    disadvantages=[],
                    research_focus="hydrogen storage",
                    key_findings=[]
                )
            else:
                knowledge = self._parse_llm_response(str(response), paper)
            
            return knowledge
            
        except Exception as e:
            print(f"Error extracting knowledge: {e}")
            return None
    
    def _create_extraction_prompt(self, paper: Dict[str, Any]) -> str:
        """Create a structured prompt for knowledge extraction"""
        
        prompt = f"""
Extract structured information about High-Entropy Alloys (HEA) for hydrogen storage from this research paper.

PAPER INFORMATION:
Title: {paper['title']}
Abstract: {paper.get('abstract', 'No abstract available')}
Journal: {paper.get('journal', 'Unknown')}
Year: {paper.get('year', 'Unknown')}

EXTRACTION INSTRUCTIONS:
Please extract the following information in JSON format. Use null for missing information:

{{
    "alloy_system": "Main alloy composition studied (e.g., AlCoCrFeNi)",
    "elements": ["List", "of", "elements", "in", "the", "alloy"],
    "crystal_structure": "Crystal structure (BCC/FCC/mixed/null)",
    "hydrogen_capacity": "Hydrogen storage capacity as number",
    "hydrogen_capacity_units": "Units (wt% or at%)",
    "absorption_temperature": "Absorption temperature in Celsius",
    "desorption_temperature": "Desorption temperature in Celsius", 
    "pressure_conditions": "Pressure in bar",
    "absorption_time": "Time for absorption in minutes",
    "desorption_time": "Time for desorption in minutes",
    "cyclic_stability": "Number of stable cycles",
    "synthesis_method": "How the alloy was made",
    "heat_treatment": "Heat treatment conditions",
    "reversibility": "reversible/irreversible/partial",
    "activation_energy": "Activation energy in kJ/mol",
    "phase_composition": ["List", "of", "phases", "present"],
    "microstructure": "Description of microstructure",
    "compared_materials": ["Materials", "compared", "against"],
    "advantages": ["Key", "advantages", "found"],
    "disadvantages": ["Limitations", "or", "disadvantages"],
    "research_focus": "Main research objective",
    "key_findings": ["Most", "important", "findings"]
}}

IMPORTANT:
- Extract only factual information from the paper
- Use null for information not mentioned
- For numerical values, extract only the number (no units in the number field)
- Be precise with alloy compositions (e.g., AlCoCrFeNi, not "multi-element")
- Focus on hydrogen storage related properties

Please provide only the JSON response:
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, paper: Dict[str, Any]) -> Optional[HEAKnowledge]:
        """Parse LLM response into HEAKnowledge object"""
        try:
            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Parse JSON
            data = json.loads(response)
            
            # Create HEAKnowledge object
            knowledge = HEAKnowledge(
                paper_id=paper.get('doi', paper.get('title', ''))[:50],
                title=paper['title'],
                alloy_system=data.get('alloy_system'),
                elements=data.get('elements', []),
                crystal_structure=data.get('crystal_structure'),
                hydrogen_capacity=data.get('hydrogen_capacity'),
                hydrogen_capacity_units=data.get('hydrogen_capacity_units'),
                absorption_temperature=data.get('absorption_temperature'),
                desorption_temperature=data.get('desorption_temperature'),
                pressure_conditions=data.get('pressure_conditions'),
                absorption_time=data.get('absorption_time'),
                desorption_time=data.get('desorption_time'),
                cyclic_stability=data.get('cyclic_stability'),
                synthesis_method=data.get('synthesis_method'),
                heat_treatment=data.get('heat_treatment'),
                reversibility=data.get('reversibility'),
                activation_energy=data.get('activation_energy'),
                phase_composition=data.get('phase_composition', []),
                microstructure=data.get('microstructure'),
                compared_materials=data.get('compared_materials', []),
                advantages=data.get('advantages', []),
                disadvantages=data.get('disadvantages', []),
                research_focus=data.get('research_focus', ''),
                key_findings=data.get('key_findings', [])
            )
            
            return knowledge
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
    
    def save_knowledge(self, knowledge_list: List[HEAKnowledge], filename: str):
        """Save extracted knowledge to JSON file"""
        knowledge_dict = [asdict(k) for k in knowledge_list]
        
        output = {
            'extraction_date': str(Path().absolute()),
            'total_knowledge_entries': len(knowledge_list),
            'knowledge': knowledge_dict
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"🧠 Knowledge saved to {filename}")
    
    def create_knowledge_summary(self, knowledge_list: List[HEAKnowledge], filename: str):
        """Create a human-readable summary of extracted knowledge"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("HIGH-ENTROPY ALLOYS KNOWLEDGE EXTRACTION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Overall statistics
            f.write(f"Total papers processed: {len(knowledge_list)}\n")
            
            # Alloy systems found
            alloy_systems = [k.alloy_system for k in knowledge_list if k.alloy_system]
            f.write(f"Unique alloy systems: {len(set(alloy_systems))}\n")
            f.write(f"Alloy systems: {list(set(alloy_systems))}\n\n")
            
            # Storage capacities
            capacities = [k.hydrogen_capacity for k in knowledge_list if k.hydrogen_capacity]
            if capacities:
                f.write(f"Hydrogen storage capacities (range): {min(capacities):.2f} - {max(capacities):.2f}\n")
                f.write(f"Average capacity: {sum(capacities)/len(capacities):.2f}\n\n")
            
            # Detailed entries
            f.write("DETAILED KNOWLEDGE ENTRIES\n")
            f.write("-"*60 + "\n\n")
            
            for i, knowledge in enumerate(knowledge_list, 1):
                f.write(f"ENTRY {i}\n")
                f.write(f"Title: {knowledge.title}\n")
                f.write(f"Alloy System: {knowledge.alloy_system}\n")
                f.write(f"Elements: {', '.join(knowledge.elements) if knowledge.elements else 'N/A'}\n")
                f.write(f"Crystal Structure: {knowledge.crystal_structure}\n")
                
                if knowledge.hydrogen_capacity:
                    f.write(f"H2 Capacity: {knowledge.hydrogen_capacity} {knowledge.hydrogen_capacity_units or ''}\n")
                
                if knowledge.synthesis_method:
                    f.write(f"Synthesis: {knowledge.synthesis_method}\n")
                
                if knowledge.key_findings:
                    f.write(f"Key Findings: {'; '.join(knowledge.key_findings)}\n")
                
                f.write("\n" + "-"*40 + "\n\n")
        
        print(f"📄 Knowledge summary saved to {filename}")

def main():
    """Main function to demonstrate knowledge extraction"""
    print("🧠 Starting HEA Knowledge Extraction")
    
    # Check if papers file exists
    papers_file = 'hea_hydrogen_papers.json'
    if not os.path.exists(papers_file):
        print(f"❌ Papers file {papers_file} not found. Run hea_literature_fetcher.py first.")
        return
    
    # Initialize extractor
    extractor = HEAKnowledgeExtractor(use_gemini=True)
    
    # Extract knowledge
    knowledge_list = extractor.extract_knowledge_from_papers(papers_file)
    
    if knowledge_list:
        # Save extracted knowledge
        extractor.save_knowledge(knowledge_list, 'hea_knowledge.json')
        extractor.create_knowledge_summary(knowledge_list, 'hea_knowledge_summary.txt')
        
        print(f"\n✅ Successfully extracted knowledge from {len(knowledge_list)} papers")
    else:
        print("❌ No knowledge extracted")

if __name__ == "__main__":
    main()
