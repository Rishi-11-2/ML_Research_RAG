"""
ArXiv ML Papers Finder
======================

This script searches arXiv for machine-learning related papers, scores them
with simple heuristics, and saves only the arXiv links (abs and PDF) for the
highest-scoring / most-relevant results.

Usage:
    python arxiv_ml_papers_finder.py

Output:
    - arxiv_ml_papers_links_only.txt  (contains one paper per line: ABS URL | PDF URL)

Notes / heuristics used to decide "good":
 - restricts to ML-relevant arXiv categories (cs.LG, cs.CV, cs.AI, stat.ML, eess.AS)
 - prefers recent papers (last 5 years) but retains notable older work
 - boosts papers mentioning top conferences/journal references in metadata
 - boosts papers whose title/abstract include strong ML keywords (transformer, diffusion, contrastive, etc.)
 - uses abstract length, author count as minor signals

This intentionally avoids external APIs and only uses the arXiv API.
"""

import requests
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import math

ARXIV_NS = '{http://www.w3.org/2005/Atom}'
ARXIV_EXT = '{http://arxiv.org/schemas/atom}'

@dataclass
class Paper:
    title: str
    abs_url: str
    pdf_url: str
    authors: List[str]
    year: int
    abstract: str
    categories: List[str]
    journal_ref: Optional[str] = None
    comment: Optional[str] = None
    score: float = 0.0


class ArxivMLFinder:
    def __init__(self):
        # categories that are strongly ML-related; others may slip through if query matches
        self.allowed_categories = set(['cs.LG', 'cs.CV', 'stat.ML', 'cs.AI', 'eess.AS'])
        # keywords that signal ML relevance/modern research
        self.strong_keywords = [
            'transformer', 'attention', 'diffusion', 'graph neural', 'gcn', 'gnn',
            'contrastive', 'self-supervised', 'semi-supervised', 'meta-learning',
            'few-shot', 'foundation model', 'large language', 'llm', 'reinforcement learning',
            'policy gradient', 'q-learning', 'object detection', 'segmentation',
            'vision', 'speech', 'nlp', 'natural language', 'representation learning'
        ]
        self.now_year = datetime.now().year

    def search_arxiv(self, query: str, max_results: int = 50) -> List[Paper]:
        """Search arXiv API and return parsed Paper objects.
        The query is used in the arXiv "all:" search field (safe for simple queries).
        """
        print(f"🔎 Searching arXiv for: '{query}' (max {max_results})")
        url = 'http://export.arxiv.org/api/query'
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            results = []
            for entry in root.findall(ARXIV_NS + 'entry'):
                title = (entry.find(ARXIV_NS + 'title').text or '').strip().replace('\n', ' ')
                abs_id = entry.find(ARXIV_NS + 'id').text.strip()
                # construct PDF url (standard arXiv pattern)
                # abs_id commonly like 'http://arxiv.org/abs/2412.09871v1'
                pdf_url = abs_id.replace('/abs/', '/pdf/')

                abstract = (entry.find(ARXIV_NS + 'summary').text or '').strip().replace('\n', ' ')
                published = entry.find(ARXIV_NS + 'published').text
                year = int(published[:4]) if published else self.now_year

                # categories: arXiv stores primary category in <arxiv:primary_category> and others in <category term=..>
                cats = []
                primary = entry.find(ARXIV_EXT + 'primary_category')
                if primary is not None and 'term' in primary.attrib:
                    cats.append(primary.attrib['term'])
                for c in entry.findall(ARXIV_NS + 'category'):
                    term = c.attrib.get('term')
                    if term and term not in cats:
                        cats.append(term)

                # authors
                authors = []
                for a in entry.findall(ARXIV_NS + 'author'):
                    name_node = a.find(ARXIV_NS + 'name')
                    if name_node is not None:
                        authors.append((name_node.text or '').strip())

                # optional arXiv metadata
                journal_ref_node = entry.find(ARXIV_EXT + 'journal_ref')
                comment_node = entry.find(ARXIV_EXT + 'comment')
                journal_ref = journal_ref_node.text.strip() if journal_ref_node is not None and journal_ref_node.text else None
                comment = comment_node.text.strip() if comment_node is not None and comment_node.text else None

                paper = Paper(
                    title=title,
                    abs_url=abs_id,
                    pdf_url=pdf_url,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    categories=cats,
                    journal_ref=journal_ref,
                    comment=comment
                )
                results.append(paper)

            print(f"✅ arXiv returned {len(results)} entries for query '{query}'")
            return results

        except Exception as e:
            print(f"❌ Error fetching from arXiv for query '{query}': {e}")
            return []

    def score_paper(self, paper: Paper) -> float:
        """Simple heuristic scoring for arXiv ML papers. Higher is better."""
        score = 0.0

        # recency: prefer last 3 years heavily, last 5 years moderately
        age = max(0, self.now_year - paper.year)
        if age <= 1:
            score += 25
        elif age <= 3:
            score += 18
        elif age <= 5:
            score += 10
        else:
            score += max(0, 5 - math.log1p(age))  # small credit for older classics

        # abstract length (very short abstracts often indicate posters or low-detail submissions)
        abs_len = len(paper.abstract.split())
        if abs_len >= 150:
            score += 8
        elif abs_len >= 80:
            score += 5

        # journal_ref or comment indicating acceptance/publication adds large boost
        if paper.journal_ref:
            score += 20
            # if journal_ref mentions known conference/journal, add more
            confs = ['neurips', 'icml', 'iclr', 'cvpr', 'iccv', 'eccv', 'aaai', 'acl', 'emnlp', 'nature', 'science']
            jr = paper.journal_ref.lower()
            for c in confs:
                if c in jr:
                    score += 12
                    break

        if paper.comment:
            com = paper.comment.lower()
            # common phrases: 'accepted at', 'to appear in', 'in proceedings of'
            if 'accepted' in com or 'to appear' in com or 'to appear in' in com or 'in proceedings' in com:
                score += 10

        # category boost if in allowed ML categories
        if any(cat in self.allowed_categories for cat in paper.categories):
            score += 8

        # strong keyword matches in title or abstract
        text = (paper.title + '\n' + paper.abstract).lower()
        kw_hits = 0
        for kw in self.strong_keywords:
            if kw in text:
                kw_hits += 1
        score += min(20, kw_hits * 6)

        # author count: collaborative papers sometimes indicate larger study
        if len(paper.authors) >= 3:
            score += 3

        return round(score, 2)

    def filter_and_rank(self, papers: List[Paper], min_score: float = 20.0) -> List[Paper]:
        """Score, filter by threshold, remove duplicates, and return sorted list."""
        print("📊 Scoring and filtering arXiv papers...")
        for p in papers:
            p.score = self.score_paper(p)

        # deduplicate by normalized title
        unique = {}
        for p in papers:
            norm = ''.join(p.title.lower().split())[:200]
            if len(norm) <= 10:
                continue
            # keep the higher-scoring duplicate
            if norm not in unique or p.score > unique[norm].score:
                unique[norm] = p

        candidates = list(unique.values())
        # filter
        filtered = [p for p in candidates if p.score >= min_score]
        filtered.sort(key=lambda x: x.score, reverse=True)

        print(f"✅ {len(filtered)} papers passed the score threshold (>= {min_score})")
        return filtered

    def save_links_only(self, papers: List[Paper], filename: str = 'paper_links.txt'):
        print(f"💾 Saving {len(papers)} arXiv links to '{filename}'")
        with open(filename, 'w', encoding='utf-8') as f:
            for p in papers:
                f.write(f"{p.pdf_url}\n")
        print("✅ Done saving links.")


def main():
    finder = ArxivMLFinder()

    # Customize queries below for the exact ML subareas you care about.
    queries = [
        'machine learning transformer',
        'deep learning neural networks',
        'representation learning',
        'self-supervised learning',
        'foundation models large language models',
        'graph neural networks',
        'diffusion models image generation',
        'contrastive learning',
        'reinforcement learning deep',
        'computer vision object detection segmentation'
    ]

    all_papers = []
    for q in queries:
        papers = finder.search_arxiv(q, max_results=50)
        all_papers.extend(papers)
        # be polite to the arXiv API
        time.sleep(1)

    # Score and filter. Tune min_score if you want more/fewer results.
    good = finder.filter_and_rank(all_papers, min_score=50.0)

    # Save only links
    finder.save_links_only(good)

    # Print a short summary
    if good:
        print('\nTop 5 saved arXiv papers:')
        for i, p in enumerate(good[:5], 1):
            print(f"{i}. {p.title[:120]} ({p.year}) — score: {p.score}")
            print(f"   {p.abs_url}")
    else:
        print('No papers passed the filter — consider lowering min_score or adding queries')


if __name__ == '__main__':
    main()
