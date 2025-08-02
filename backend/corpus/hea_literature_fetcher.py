"""
High-Entropy Alloys Literature Fetcher
Specialized fetcher for HEA hydrogen storage research papers from ScienceDirect
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pybliometrics.sciencedirect import ScienceDirectSearch
from pybliometrics.utils.startup import init

# Initialize pybliometrics
init()

@dataclass
class HEAPaper:
    """Data class for High-Entropy Alloy research papers"""
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    doi: str
    url: str
    keywords: List[str]
    
    # HEA-specific fields
    alloy_compositions: List[str] = None
    storage_capacity: Optional[float] = None
    temperature_range: Optional[str] = None
    pressure_conditions: Optional[str] = None
    
    def __post_init__(self):
        if self.alloy_compositions is None:
            self.alloy_compositions = []
        if self.keywords is None:
            self.keywords = []

class HEALiteratureFetcher:
    """
    Specialized fetcher for High-Entropy Alloys hydrogen storage literature
    """
    
    def __init__(self):
        self.hea_queries = {
            'hea_hydrogen': 'title("high entropy alloy") AND title("hydrogen")',
            'hea_storage': 'title("high entropy alloy") AND title("storage")',
            'multicomponent_hydrogen': 'title("multicomponent alloy") AND title("hydrogen")',
            'hea_hydride': 'title("high entropy") AND title("hydride")',
            'compositionally_complex': 'title("compositionally complex alloy") AND title("hydrogen")',
        }
        
        self.element_patterns = [
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zr', 'Nb', 'Mo',
            'Hf', 'Ta', 'W', 'Al', 'Mg', 'Ca', 'Li', 'Na', 'K'
        ]
    
    def fetch_hea_papers(self, max_papers_per_query: int = 50) -> List[HEAPaper]:
        """
        Fetch HEA hydrogen storage papers from all queries
        
        Args:
            max_papers_per_query: Maximum papers to fetch per query
            
        Returns:
            List of HEAPaper objects
        """
        all_papers = []
        paper_dois = set()  # To avoid duplicates
        
        for query_name, query in self.hea_queries.items():
            print(f"\n🔍 Fetching papers for: {query_name}")
            print(f"Query: {query}")
            
            try:
                search = ScienceDirectSearch(query, count=max_papers_per_query)
                
                if not search.results:
                    print(f"❌ No results for {query_name}")
                    continue
                
                print(f"✅ Found {len(search.results)} papers")
                
                for result in search.results:
                    # Skip duplicates based on DOI
                    doi = getattr(result, 'doi', '') or ''
                    if doi and doi in paper_dois:
                        continue
                    
                    paper = self._create_hea_paper(result)
                    if paper:
                        all_papers.append(paper)
                        if doi:
                            paper_dois.add(doi)
                
            except Exception as e:
                print(f"❌ Error fetching {query_name}: {e}")
                continue
        
        print(f"\n📚 Total unique papers fetched: {len(all_papers)}")
        return all_papers
    
    def _create_hea_paper(self, result) -> Optional[HEAPaper]:
        """Convert ScienceDirect result to HEAPaper object"""
        try:
            # Extract basic information
            title = result.title or "No title"
            abstract = getattr(result, 'abstract', '') or ''
            
            # Parse authors (fix the character splitting issue)
            authors_raw = getattr(result, 'authors', []) or []
            authors = self._parse_authors(authors_raw)
            
            journal = getattr(result, 'publicationName', 'Unknown Journal')
            year = self._extract_year(getattr(result, 'coverDate', ''))
            doi = getattr(result, 'doi', '') or ''
            url = getattr(result, 'link', '') or ''
            
            # Extract HEA-specific information from title and abstract
            text_content = f"{title} {abstract}"  # Don't convert to lowercase yet
            alloy_compositions = self._extract_alloy_compositions(text_content)
            keywords = self._extract_hea_keywords(text_content.lower())  # Keywords can be lowercase
            
            return HEAPaper(
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                doi=doi,
                url=url,
                keywords=keywords,
                alloy_compositions=alloy_compositions
            )
            
        except Exception as e:
            print(f"Error creating paper object: {e}")
            return None
    
    def _parse_authors(self, authors_raw) -> List[str]:
        """Parse author names properly"""
        if not authors_raw:
            return []
        
        # If authors_raw is a string that got split into characters
        if isinstance(authors_raw, list) and len(authors_raw) > 10:
            # Likely character-split, try to rejoin
            author_string = ''.join(authors_raw)
            # Split by common separators
            authors = [name.strip() for name in author_string.replace(';', ',').split(',') if name.strip()]
            return authors[:10]  # Limit to first 10 authors
        
        # If it's already a proper list of names
        if isinstance(authors_raw, list):
            return [str(author).strip() for author in authors_raw if str(author).strip()][:10]
        
        # If it's a string
        if isinstance(authors_raw, str):
            return [name.strip() for name in authors_raw.replace(';', ',').split(',') if name.strip()][:10]
        
        return []
    
    def _extract_year(self, date_string: str) -> str:
        """Extract year from date string"""
        if not date_string:
            return "Unknown"
        try:
            return date_string.split('-')[0]
        except:
            return "Unknown"
    
    def _extract_alloy_compositions(self, text: str) -> List[str]:
        """Extract potential alloy compositions from text"""
        compositions = []
        
        # Look for common HEA composition patterns
        import re
        
        # Pattern for compositions like "TiVCrMnFe", "AlCoCrFeNi", "FeCoNiCuZn", etc.
        composition_patterns = [
            # Match sequences of 3+ chemical elements (like TiVCrMnFe, AlCoCrFeNi)
            r'\b(?:[A-Z][a-z]?){3,}(?=\s|$|[^a-zA-Z])',
            # Match explicit alloy names with subscripts (like Al0.35CoCrFeNi)
            r'\b[A-Z][a-z]?(?:[0-9.]+)?(?:[A-Z][a-z]?(?:[0-9.]+)?){2,}(?=\s|alloy|HEA|$|[^a-zA-Z0-9.])',
            # Match compositions with separators (Ti-V-Cr-Mn-Fe)
            r'\b[A-Z][a-z]?(?:[-\s]+[A-Z][a-z]?){2,}(?=\s|alloy|HEA|$)',
        ]
        
        for pattern in composition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the match and validate it contains known elements
                clean_match = re.sub(r'[^A-Za-z0-9.]', '', match)
                if len(clean_match) >= 4 and self._contains_valid_elements(clean_match):
                    compositions.append(clean_match)
        
        return list(set(compositions))  # Remove duplicates
    
    def _contains_valid_elements(self, composition: str) -> bool:
        """Check if composition contains valid chemical elements"""
        import re
        
        common_elements = [
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Al', 'Nb', 'Ta',
            'Zr', 'Hf', 'Mo', 'W', 'Pt', 'Pd', 'Ru', 'Rh', 'Ir', 'Au', 'Ag', 'Mg',
            'Ca', 'Sr', 'Ba', 'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd'
        ]
        
        # Remove numbers and dots
        elements_only = re.sub(r'[0-9.]', '', composition)
        
        # Count how many valid elements we can find
        found_elements = 0
        i = 0
        while i < len(elements_only):
            # Try 2-letter element first
            if i + 1 < len(elements_only):
                two_letter = elements_only[i:i+2]
                if two_letter in common_elements:
                    found_elements += 1
                    i += 2
                    continue
            
            # Try 1-letter element
            one_letter = elements_only[i:i+1]
            if one_letter in common_elements:
                found_elements += 1
                i += 1
            else:
                i += 1
        
        return found_elements >= 3  # At least 3 valid elements for HEA
    
    def _extract_hea_keywords(self, text: str) -> List[str]:
        """Extract HEA-related keywords from text"""
        hea_keywords = [
            'high entropy alloy', 'hea', 'multicomponent alloy',
            'compositionally complex alloy', 'high entropy hydride',
            'hydrogen storage', 'hydride', 'dehydrogenation',
            'hydrogenation', 'storage capacity', 'kinetics',
            'thermodynamics', 'phase structure', 'crystalline',
            'bcc', 'fcc', 'solid solution'
        ]
        
        found_keywords = []
        for keyword in hea_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def save_papers_json(self, papers: List[HEAPaper], filename: str):
        """Save papers to JSON file"""
        papers_dict = [asdict(paper) for paper in papers]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'fetch_date': datetime.now().isoformat(),
                'total_papers': len(papers),
                'papers': papers_dict
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Papers saved to {filename}")
    
    def save_papers_text(self, papers: List[HEAPaper], filename: str):
        """Save papers to readable text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("HIGH-ENTROPY ALLOYS FOR HYDROGEN STORAGE - LITERATURE REVIEW\n")
            f.write("="*80 + "\n")
            f.write(f"Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Papers: {len(papers)}\n\n")
            
            for i, paper in enumerate(papers, 1):
                f.write(f"PAPER {i}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Title: {paper.title}\n")
                f.write(f"Authors: {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}\n")
                f.write(f"Journal: {paper.journal} ({paper.year})\n")
                f.write(f"DOI: {paper.doi}\n")
                
                if paper.alloy_compositions:
                    f.write(f"Potential Alloy Compositions: {', '.join(paper.alloy_compositions)}\n")
                
                if paper.keywords:
                    f.write(f"HEA Keywords: {', '.join(paper.keywords)}\n")
                
                if paper.abstract:
                    f.write(f"Abstract: {paper.abstract[:300]}{'...' if len(paper.abstract) > 300 else ''}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"📄 Readable summary saved to {filename}")

def main():
    """Main function to demonstrate HEA literature fetching"""
    print("🚀 Starting HEA Literature Fetcher")
    print("Focus: High-Entropy Alloys for Hydrogen Storage")
    
    fetcher = HEALiteratureFetcher()
    
    # Fetch papers
    papers = fetcher.fetch_hea_papers(max_papers_per_query=30)
    
    if papers:
        # Save in different formats
        fetcher.save_papers_json(papers, 'hea_hydrogen_papers.json')
        fetcher.save_papers_text(papers, 'hea_hydrogen_papers.txt')
        
        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"Total papers: {len(papers)}")
        
        # Count by year
        year_counts = {}
        for paper in papers:
            year = paper.year
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"Year distribution: {dict(sorted(year_counts.items()))}")
        
        # Most common journals
        journal_counts = {}
        for paper in papers:
            journal = paper.journal
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        
        top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top journals: {top_journals}")
        
    else:
        print("❌ No papers found")

if __name__ == "__main__":
    main()
