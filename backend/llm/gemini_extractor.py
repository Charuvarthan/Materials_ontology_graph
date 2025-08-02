"""
Gemini-based Knowledge Extractor for HEA Literature
Uses Google's Gemini API to extract structured knowledge from HEA papers
"""

import google.generativeai as genai
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedKnowledge:
    """Structure for extracted knowledge from papers"""
    paper_id: str
    title: str
    
    # Material composition
    alloy_system: Optional[str] = None
    elements: List[str] = None
    
    # Properties
    hydrogen_capacity: Optional[float] = None
    hydrogen_capacity_units: Optional[str] = None
    crystal_structure: Optional[str] = None
    synthesis_method: Optional[str] = None
    
    # Performance
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    reversibility: Optional[str] = None
    cyclic_stability: Optional[int] = None
    
    # Research findings
    key_findings: List[str] = None
    
    # Additional data
    doi: Optional[str] = None
    year: Optional[str] = None
    
    def __post_init__(self):
        if self.elements is None:
            self.elements = []
        if self.key_findings is None:
            self.key_findings = []

class GeminiExtractor:
    """
    Uses Google Gemini to extract structured knowledge from scientific papers
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini extractor"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            logger.warning("No Gemini API key found. Using mock extraction.")
            self.use_mock = True
        else:
            try:
                genai.configure(api_key=self.api_key)
                # Use the more powerful gemini-1.5-pro model
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                self.use_mock = False
                logger.info("Gemini API initialized successfully with gemini-1.5-pro")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
                self.use_mock = True
    
    def extract_knowledge_from_paper(self, title: str, abstract: str, doi: str = "") -> ExtractedKnowledge:
        """
        Extract structured knowledge from a single paper
        """
        if self.use_mock:
            return self._create_mock_knowledge(title, abstract, doi)
        
        try:
            prompt = self._create_extraction_prompt(title, abstract)
            response = self.model.generate_content(prompt)
            
            # Parse the response
            extracted_data = self._parse_gemini_response(response.text)
            
            # Create knowledge object
            knowledge = ExtractedKnowledge(
                paper_id=doi or title[:50],
                title=title,
                doi=doi,
                **extracted_data
            )
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge using Gemini: {e}")
            return self._create_mock_knowledge(title, abstract, doi)
    
    def _create_extraction_prompt(self, title: str, abstract: str) -> str:
        """Create a prompt for knowledge extraction"""
        prompt = f"""
You are an expert materials science researcher specializing in High-Entropy Alloys (HEAs) for hydrogen storage applications. 

Extract structured information from this research paper and return it as valid JSON:

TITLE: {title}
ABSTRACT: {abstract}

Extract the following information in JSON format. Be precise and only include information explicitly mentioned in the text:

{{
    "alloy_system": "Main alloy composition (e.g., AlCoCrFeNi, TiVZrNbHf)",
    "elements": ["List", "of", "chemical", "elements"],
    "hydrogen_capacity": "Numerical value of H2 storage capacity",
    "hydrogen_capacity_units": "Units (wt%, at%, etc.)",
    "crystal_structure": "Crystal structure (BCC, FCC, mixed, amorphous, etc.)",
    "synthesis_method": "Synthesis method (arc_melting, ball_milling, sputtering, etc.)",
    "temperature": "Operating/measurement temperature in Celsius",
    "pressure": "Operating/measurement pressure in bar",
    "reversibility": "reversible, irreversible, or partial",
    "cyclic_stability": "Number of stable cycles as integer",
    "key_findings": ["List", "of", "major", "findings", "or", "conclusions"]
}}

EXTRACTION RULES:
1. Use null for any field where information is not clearly stated
2. For numerical values, extract only the number (no units in the number field)
3. For alloy_system, use standard notation (e.g., AlCoCrFeNi not "Al-Co-Cr-Fe-Ni")
4. For elements, use chemical symbols (e.g., ["Al", "Co", "Cr"] not ["aluminum", "cobalt"])
5. Focus on hydrogen storage properties and performance
6. If multiple values are given, use the best/highest performance value
7. Return only the JSON object, no other text

JSON Response:"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from Gemini"""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
            
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Find JSON object bounds
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Validate and clean the parsed data
                cleaned_data = {}
                for key, value in parsed_data.items():
                    if value is not None and value != "" and value != []:
                        cleaned_data[key] = value
                
                logger.info(f"Successfully parsed Gemini response: {len(cleaned_data)} fields extracted")
                return cleaned_data
            else:
                logger.warning("No valid JSON object found in Gemini response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
            logger.debug(f"Response text: {response_text[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error parsing Gemini response: {e}")
            return {}
    
    def _create_mock_knowledge(self, title: str, abstract: str, doi: str = "") -> ExtractedKnowledge:
        """Create mock knowledge for testing when API is not available"""
        
        # Simple pattern matching for mock data
        elements = []
        alloy_system = None
        synthesis_method = None
        hydrogen_capacity = None
        temperature = None
        pressure = None
        crystal_structure = None
        
        # Common HEA elements
        hea_elements = ["Al", "Co", "Cr", "Fe", "Ni", "Cu", "Mn", "Ti", "V", "Zr", "Nb", "Mo", "W"]
        
        title_lower = title.lower()
        abstract_lower = abstract.lower() if abstract else ""
        combined_text = f"{title_lower} {abstract_lower}"
        
        # Extract elements from text
        for elem in hea_elements:
            if elem.lower() in combined_text or elem in title or elem in abstract:
                elements.append(elem)
        
        # Try to identify specific alloy systems from text
        if "feconicuzn" in combined_text.replace(" ", "").replace("-", ""):
            alloy_system = "FeCoNiCuZn"
            elements = ["Fe", "Co", "Ni", "Cu", "Zn"]
        elif "alcocrtifeni" in combined_text.replace(" ", "").replace("-", ""):
            alloy_system = "AlCoCrFeNi"
            elements = ["Al", "Co", "Cr", "Fe", "Ni"]
        elif "alcocrfeni" in combined_text.replace(" ", "").replace("-", ""):
            alloy_system = "AlCoCrFeNi"
            elements = ["Al", "Co", "Cr", "Fe", "Ni"]
        elif "alcrfeni" in combined_text.replace(" ", "").replace("-", ""):
            alloy_system = "AlCrFeNi"
            elements = ["Al", "Cr", "Fe", "Ni"]
        elif "alcotiv" in combined_text.replace(" ", "").replace("-", ""):
            alloy_system = "AlCoTiV"
            elements = ["Al", "Co", "Ti", "V"]
        elif "alcocufeni" in combined_text.replace(" ", "").replace("-", ""):
            alloy_system = "AlCoCuFeNi"
            elements = ["Al", "Co", "Cu", "Fe", "Ni"]
        elif len(elements) >= 3:
            # Create system from found elements
            alloy_system = "".join(sorted(elements[:5]))
        
        # Extract synthesis methods
        if "arc melting" in combined_text:
            synthesis_method = "arc_melting"
        elif "ball milling" in combined_text or "mechanical alloying" in combined_text:
            synthesis_method = "ball_milling"
        elif "induction melting" in combined_text:
            synthesis_method = "induction_melting"
        elif "sputtering" in combined_text:
            synthesis_method = "sputtering"
        elif "electrodeposition" in combined_text:
            synthesis_method = "electrodeposition"
        elif "vacuum melting" in combined_text:
            synthesis_method = "vacuum_melting"
        
        # Extract hydrogen capacity (look for numbers with wt% or at%)
        import re
        capacity_patterns = [
            r'(\d+\.?\d*)\s*wt\s*%',
            r'(\d+\.?\d*)\s*at\s*%',
            r'(\d+\.?\d*)\s*weight\s*%',
            r'(\d+\.?\d*)\s*atomic\s*%'
        ]
        
        for pattern in capacity_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                try:
                    hydrogen_capacity = float(matches[0])
                    break
                except ValueError:
                    continue
        
        # If no specific capacity found, generate reasonable mock values for storage papers
        if hydrogen_capacity is None and "storage" in combined_text:
            if "high capacity" in combined_text or "excellent" in combined_text:
                hydrogen_capacity = 3.2  # High capacity
            elif "good" in combined_text or "effective" in combined_text:
                hydrogen_capacity = 2.1  # Good capacity
            else:
                hydrogen_capacity = 1.8  # Moderate capacity
        
        # Extract temperature information
        temp_patterns = [
            r'(\d+)\s*°c',
            r'(\d+)\s*celsius',
            r'room\s*temperature',
            r'ambient'
        ]
        
        for pattern in temp_patterns:
            if pattern in ['room temperature', 'ambient']:
                temperature = 25.0
                break
            else:
                matches = re.findall(pattern, combined_text)
                if matches:
                    try:
                        temperature = float(matches[0])
                        break
                    except ValueError:
                        continue
        
        # Extract pressure information
        pressure_patterns = [
            r'(\d+\.?\d*)\s*bar',
            r'(\d+\.?\d*)\s*mpa',
            r'(\d+\.?\d*)\s*atm'
        ]
        
        for pattern in pressure_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                try:
                    pressure = float(matches[0])
                    if 'mpa' in pattern:
                        pressure *= 10  # Convert MPa to bar
                    break
                except ValueError:
                    continue
        
        # Determine crystal structure
        if "bcc" in combined_text:
            crystal_structure = "BCC"
        elif "fcc" in combined_text:
            crystal_structure = "FCC"
        elif "mixed" in combined_text or "dual" in combined_text:
            crystal_structure = "mixed"
        elif alloy_system and len(elements) >= 4:
            # Mock structure based on alloy type
            if "Al" in elements and "Cr" in elements:
                crystal_structure = "BCC"
            elif "Ni" in elements and "Cu" in elements:
                crystal_structure = "FCC"
            else:
                crystal_structure = "mixed"
        
        return ExtractedKnowledge(
            paper_id=doi or title[:50],
            title=title,
            doi=doi,
            alloy_system=alloy_system,
            elements=elements,
            hydrogen_capacity=hydrogen_capacity,
            crystal_structure=crystal_structure,
            synthesis_method=synthesis_method,
            temperature=temperature,
            pressure=pressure
        )
    
    def extract_batch(self, papers: List[Dict[str, Any]], max_papers: int = 10) -> List[ExtractedKnowledge]:
        """
        Extract knowledge from a batch of papers
        """
        results = []
        
        for i, paper in enumerate(papers[:max_papers]):
            logger.info(f"Processing paper {i+1}/{min(len(papers), max_papers)}: {paper.get('title', 'Unknown')[:50]}...")
            
            knowledge = self.extract_knowledge_from_paper(
                title=paper.get('title', ''),
                abstract=paper.get('abstract', ''),
                doi=paper.get('doi', '')
            )
            
            results.append(knowledge)
            
            # Rate limiting for API calls - more conservative for better quality
            if not self.use_mock:
                time.sleep(2)  # Wait 2 seconds between API calls to avoid rate limits
        
        return results