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
import re

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


    def count_syllables(self,word):
        """Simple heuristic to count syllables in a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    def flesch_kincaid_grade(self,text):
        """Calculate Flesch-Kincaid grade level for readability."""
        sentences = re.split(r'[.!?]+', text)
        num_sentences = len([s for s in sentences if s.strip()])
        if num_sentences == 0:
            return 0
        words = re.findall(r'\w+', text)
        num_words = len(words)
        if num_words == 0:
            return 0
        num_syllables = sum(self.count_syllables(word) for word in words)
        asl = num_words / num_sentences  # average sentence length
        asw = num_syllables / num_words  # average syllables per word
        grade = 0.39 * asl + 11.8 * asw - 15.59
        return grade

    def search_arxiv(self, query: str, max_results: int = 100) -> List[Paper]:
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
        """Improved heuristic scoring for arXiv ML papers to assess quality more accurately. Higher is better."""
        score = 0.0

        # Recency: exponential decay for better granularity
        age = max(0, self.now_year - paper.year)
        score += 25 * math.exp(-0.5 * age)  # decays slower for recent, faster for old

        # Abstract length and quality
        abs_text = paper.abstract
        abs_words = len(abs_text.split())
        if abs_words >= 200:
            score += 10
        elif abs_words >= 150:
            score += 8
        elif abs_words >= 100:
            score += 5

        # Readability: prefer readable abstracts (grade 12-18 for research papers)
        readability = self.flesch_kincaid_grade(abs_text)
        if 12 <= readability <= 18:
            score += 8
        elif 10 <= readability <= 20:
            score += 4

        # Journal ref or publication boost, expanded conference list
        confs = [
            'neurips', 'icml', 'iclr', 'cvpr', 'iccv', 'eccv', 'aaai', 'acl', 'emnlp',
            'nature', 'science', 'kdd', 'ijcai', 'sigir', 'www', 'icra', 'iros', 'rss',
            'colt', 'uai', 'aistats'
        ]
        if paper.journal_ref:
            score += 20
            jr_lower = paper.journal_ref.lower()
            for c in confs:
                if c in jr_lower:
                    score += 15  # increased boost for top venues
                    break

        # Comment indicating acceptance or additional quality signals
        if paper.comment:
            com_lower = paper.comment.lower()
            acceptance_phrases = ['accepted', 'to appear', 'in proceedings', 'camera-ready', 'oral', 'spotlight', 'best paper']
            if any(phrase in com_lower for phrase in acceptance_phrases):
                score += 12  # increased from 10

        # Category boost, assuming self.allowed_categories is list of ML cats like 'cs.LG', 'stat.ML'
        if any(cat in self.allowed_categories for cat in paper.categories):
            score += 8

        # Strong keyword matches, with expanded list if possible
        text_lower = (paper.title + '\n' + abs_text).lower()
        # Assume self.strong_keywords exists; add more for quality signals
        additional_keywords = ['novel', 'state-of-the-art', 'sota', 'benchmark', 'large-scale', 'reproducible', 'open-source', 'empirical study', 'theoretical analysis']
        all_keywords = self.strong_keywords + additional_keywords
        kw_hits = sum(1 for kw in all_keywords if kw in text_lower)
        score += min(25, kw_hits * 5)  # adjusted for more hits

        # Reproducibility signals
        repro_signals = ['github', 'code available', 'repository', 'supplementary material', 'dataset released']
        if any(sig in text_lower or (paper.comment and sig in com_lower) for sig in repro_signals):
            score += 10

        # Author count: more collaborative, but cap
        num_authors = len(paper.authors)
        if num_authors >= 5:
            score += 5
        elif num_authors >= 3:
            score += 3

        # Survey or review boost
        if 'survey' in text_lower or 'review' in text_lower:
            score += 10  # often high-quality overviews

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
        'computer vision object detection segmentation',
        'Supervised learning',
        'Unsupervised learning',
        'Semi-supervised learning',
        'Self-supervised learning',
        'Representation learning',
        'Feature learning',
        'Contrastive learning',
        'Metric learning',
        'Embedding learning',
        'Deep learning',
        'Convolutional neural networks',
        'Recurrent neural networks',
        'Transformers',
        'Attention mechanisms',
        'Graph neural networks',
        'Variational autoencoders',
        'Generative adversarial networks',
        'Normalizing flows',
        'Diffusion models',
        'Autoregressive models',
        'Sequence-to-sequence models',
        'Language modeling',
        'Multilingual modeling',            
        'Few-shot learning',
        'Zero-shot learning',
        'Meta-learning',
        'Transfer learning',
        'Domain adaptation',
        'Domain generalization',
        'Continual learning',
        'Lifelong learning',
        'Reinforcement learning',
        'Deep reinforcement learning',
        'Inverse reinforcement learning',
        'Imitation learning',
        'Multi-agent reinforcement learning',
        'Causal inference in ML',
        'Bayesian deep learning',
        'Probabilistic modeling',
        'Uncertainty estimation',
        'Out-of-distribution detection',
        'Adversarial attacks',
        'Adversarial defenses',
        'Adversarial robustness',
        'Fairness in ML',
        'Bias mitigation',
        'Explainability',
        'Interpretability',
        'Model probing',
        'Rationale generation',
        'Optimization algorithms',
        'Adaptive optimizers',
        'Second-order optimization',
        'Learning rate schedules',
        'Regularization techniques',
        'Normalization methods',
        'Dropout techniques',
        'Sparse models',
        'Neural architecture search',
        'AutoML',
        'Model compression',
        'Quantization',
        'Pruning',
        'Knowledge distillation',
        'Efficient transformer architectures',
        'Sparse attention',
        'Long-context modeling',
        'Memory-augmented networks',
        'Scalable distributed training',
        'Mixed-precision training',
        'Curriculum learning',
        'Active learning',
        'Data augmentation',
        'Synthetic data generation',
        'Data labeling strategies',
        'Human-in-the-loop learning',
        'Benchmarking and evaluation metrics',
        'Dataset curation',
        'Dataset bias analysis',
        'Privacy-preserving ML' ,
        'Differential privacy',
        'Federated learning',
        'Representation disentanglement',
        'Information-theoretic approaches',
        'Graph representation learning',
        'Temporal modeling',
        'Time-series forecasting',
        'Multimodal learning',
        'Cross-modal retrieval',
        'Vision-language modeling',
        'Speech recognition',
        'Speech synthesis',
        'Audio representation learning',
        'Video understanding',
        'Image classification',
        'Object detection',
        'Image segmentation',
        'Anomaly detection',
        'Survival analysis'

    ]

    all_papers = []
    for q in queries:
        papers = finder.search_arxiv(q)
        all_papers.extend(papers)
        # time.sleep(1)

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
