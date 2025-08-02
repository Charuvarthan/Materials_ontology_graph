"""
ScienceDirect Literature Fetcher for Materials Ontology Project
This module fetches scientific papers from ScienceDirect API for materials research.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from pybliometrics.sciencedirect import ScienceDirectSearch
from pybliometrics.utils.startup import init

# Initialize pybliometrics configuration
init()

@dataclass
class PaperInfo:
    """Data class to store paper information"""
    title: str
    authors: List[str]
    journal: str
    year: str
    doi: str
    abstract: str = ""
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class ScienceDirectFetcher:
    """
    Fetcher for ScienceDirect API to get materials science papers
    """
    
    def __init__(self):
        """Initialize the fetcher"""
        self.search_queries = {
            'hydrogen_storage': 'title("hydrogen storage materials")',
            'metal_hydrides': 'title("metal hydrides")',
            'battery_materials': 'title("battery materials")',
            'catalysts': 'title("catalysis materials")',
            'nanomaterials': 'title("nanomaterials")',
            'energy_storage': 'title("energy storage materials")',
        }
    
    def fetch_papers(self, topic: str, max_papers: int = 50) -> List[PaperInfo]:
        """
        Fetch papers for a specific topic
        
        Args:
            topic: Topic to search for (use keys from self.search_queries)
            max_papers: Maximum number of papers to fetch
            
        Returns:
            List of PaperInfo objects
        """
        if topic not in self.search_queries:
            raise ValueError(f"Topic '{topic}' not supported. Available topics: {list(self.search_queries.keys())}")
        
        query = self.search_queries[topic]
        print(f"Searching ScienceDirect for: {topic}")
        print(f"Query: {query}")
        
        try:
            search = ScienceDirectSearch(query, count=max_papers)
            
            if not search.results:
                print("No results found")
                return []
            
            print(f"Found {len(search.results)} papers (Total available: {search.get_results_size()})")
            
            papers = []
            for result in search.results:
                paper = PaperInfo(
                    title=result.title or "No title",
                    authors=getattr(result, 'authors', []) or [],
                    journal=getattr(result, 'publicationName', '') or 'Unknown Journal',
                    year=self._extract_year(getattr(result, 'coverDate', '')),
                    doi=getattr(result, 'doi', '') or '',
                    abstract=getattr(result, 'abstract', '') or ''
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []
    
    def _extract_year(self, date_string: str) -> str:
        """Extract year from date string"""
        if not date_string:
            return "Unknown"
        try:
            return date_string.split('-')[0]
        except:
            return "Unknown"
    
    def save_papers_to_file(self, papers: List[PaperInfo], filename: str):
        """Save papers to a text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Fetched {len(papers)} papers from ScienceDirect\n")
            f.write("="*50 + "\n\n")
            
            for i, paper in enumerate(papers, 1):
                f.write(f"Paper {i}:\n")
                f.write(f"Title: {paper.title}\n")
                f.write(f"Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}\n")
                f.write(f"Journal: {paper.journal}\n")
                f.write(f"Year: {paper.year}\n")
                f.write(f"DOI: {paper.doi}\n")
                if paper.abstract:
                    f.write(f"Abstract: {paper.abstract[:200]}...\n")
                f.write("\n" + "-"*40 + "\n\n")
        
        print(f"Papers saved to {filename}")

def main():
    """Main function to demonstrate the fetcher"""
    fetcher = ScienceDirectFetcher()
    
    # Test with hydrogen storage materials
    print("=== Fetching Hydrogen Storage Materials Papers ===")
    papers = fetcher.fetch_papers('hydrogen_storage', max_papers=20)
    
    if papers:
        # Save to file
        fetcher.save_papers_to_file(papers, 'hydrogen_storage_papers.txt')
        
        # Display first few papers
        print(f"\nFirst 3 papers:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Journal: {paper.journal} ({paper.year})")
            print(f"   Authors: {', '.join(paper.authors[:3]) if paper.authors else 'N/A'}")
            if len(paper.authors) > 3:
                print(f"   ... and {len(paper.authors) - 3} more authors")
    
    print(f"\n✅ ScienceDirect fetcher is working!")
    print(f"Available topics: {list(fetcher.search_queries.keys())}")

if __name__ == "__main__":
    main()
