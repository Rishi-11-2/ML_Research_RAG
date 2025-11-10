import os
import numpy as np
from typing import List, Tuple
import re
# import chromadb
import numpy as np

from dotenv import load_dotenv
from litellm import completion
from sentence_transformers import SentenceTransformer, CrossEncoder , util
from qdrant_client import QdrantClient , models

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "document_chunks")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

#--------------------------
# -----------------------
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENROUTER_API_KEY"] = OPEN_ROUTER_API_KEY



embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



# -----------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a and b are 1D arrays
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: 
        return 0.0
    return float(np.dot(a, b) / denom)

#-------------------------

def cross_encoder_rerank(query: str, docs: List, top_k: int = 5) -> List[dict]:
    """
    Accepts either:
      - docs: List[str]
      - docs: List[dict] where each dict has a 'text' key (as returned by call_vector_db)
    Returns list of dicts: {"id": None, "text": <doc string>, "score": <float>, "metadata": {}}
    """



    pairs = [[query, d.get("text")] for d in docs]
    scores = cross_encoder.predict(pairs)  # numpy array or list of floats
    scores = scores.tolist()
    results = []
    for (d, score) in zip(docs,scores):
        results.append({
            "id": d.get("id"),
            "text": d.get("text"),
            "score": float(score) if score is not None else None,
            "metadata": d.get("metadata"),
            "source": d.get("source")
        })
    return results


def call_vector_db(query: str, k: int) -> List[dict]:
    """
    

    uses the dense and sparse vectors, with their results combined
    using the Reciprocal Rank Fusion.
    """
    query_vec = embedder.encode([query])[0]
    client = QdrantClient(url=QDRANT_URL, api_key=(QDRANT_API_KEY.strip() ), prefer_grpc=False)
    # Search Qdrant
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec, 
        limit=k,
        with_vectors=False,

    )
    out = []

    points_list = getattr(results, "points", None) or results

    for hit in points_list:
        # --- support object-like hits (ScoredPoint) ---
        payload = getattr(hit, "payload", None) or {}
        score = getattr(hit, "score", None)

        text = payload.get("text")
        meta = {k: v for k, v in payload.items() if k not in ("text",  "source", "_stable_id")}

        source = payload.get("source")
        score_f = float(score) 


        # print(source)
        pid_str = payload.get("_stable_id")

        # print(pid_str, meta,score_f)
        out.append({
            "id": pid_str,
            "source":source,
            "text": text,
            "score": score_f,
            "metadata": meta,
        })

    return out

#---------------------------


def llm_call(prompt: str) -> str:
    messages = [
    {"role": "system", "content": ''' You are an extremely intelligent assistant that answers user questions by retrieving and synthesizing evidence from the context provided.

Follow these rules exactly for every response:

1. Evidence and factual rules
- Only use the retrieved context for factual claims.
- Every factual statement must be followed by an inline numeric citation that refers to a retrieved source, like this: [1] or [2,3].
- If no context supports a claim, reply:
  "I am sorry but can you provide me with a better context so as to answer your question."
  Do not invent or assume facts.
- Never rely on internal or background knowledge unless the user explicitly asks for it.
- If the user asks for opinions or general background, clearly label it as:
  "Opinion (not evidence-backed): …"

2. Citing evidence
- Each factual sentence MUST have an inline citation number (e.g., This is true.[1])
- After the answer, always include a section titled **Sources:**
- Under Sources, list every cited document in numeric order.
- Each entry must contain only the filename of the document as given in the retrieved context.
  Example:
  Sources:
  [1]: "U-Net_Segmentation_Architecture.pdf"
  [2]: "Medical_Image_Segmentation_With_U-Net.pdf"

3. Citation rules
- Cite only documents returned by the retriever.
- If multiple retrieved documents support a claim, cite up to three (e.g., [2,5,7]).
- Do not print scores, embeddings, or passage text.

4. Answer composition
- Begin with a one-sentence summary.
- Add supporting sentences or short bullet points, each with inline citations.
- If evidence conflicts, state the disagreement and cite both sides.


6. Formatting
- Use `[number]` exactly for inline citations.
- Use fenced code blocks for code, tables, or long data.
- Write naturally. Do not use phrases like "given the context."
- ALWAYS end your answer with a "Sources:" section listing filenames.
Example output format:
U-Net is an encoder-decoder network for pixel-wise image prediction.[1]
Skip connections connect encoder and decoder layers to preserve spatial detail.[2,3]
It is widely used for medical image segmentation and diffusion models.[4]

Sources:
[1]: "U-Net_Architecture_Overview.pdf"
[2]: "Deep_Learning_For_Segmentation.pdf"
[3]: "Medical_Segmentation_With_U-Net.pdf"
[4]: "U-Net_Variants_And_Applications.pdf"
MAKE SURE EVERY SOURCE IS USED ONCE
PLEASE  MAKE SURE THAT THE SOURCE IS READABLE AND THERE IS '_' BETWEEN EVERY WORD OF THE SOURCE '''},
    {"role": "user", "content": prompt}
    ]
    resp = completion(
        model="openrouter/openai/gpt-oss-20b:free",
        messages=messages,
    )
    return resp.choices[0].message.content

#------------------------------


#JUST FOR Testing 
def rag_query(query: str, final_k: int = 3, candidate_k: int = 50) -> str:
    """
    1) get candidate_k candidates from Chroma
    2) rerank them (embedding or cross-encoder)
    3) build context using top final_k
    4) send to LLM
    """
    # fetch many candidates from chroma/qdrant (bigger pool -> better reranking)
    candidates = call_vector_db(query, candidate_k)
    if not candidates:
        raise ValueError("No documents returned from Qdrant DB")

    # rerank
    # print(candidat)
    reranked = cross_encoder_rerank(query, candidates, top_k=final_k)

    # Build context text -> preserve ordering and include score for debugging if wanted
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(reranked))

   
    context = f"context:{context}, query : {query}"
    resp = llm_call(context)
    return resp

if __name__ == "__main__":
    # q = "How can we explain the whole process of FDB mathematically?"
    q = "what is U-NET?"
    print(rag_query(q, final_k=3, candidate_k=50))


# points=[ScoredPoint(id='7e911e27-7047-543a-afb1-f3ea08414398', version=452, score=0.7334566, payload={'source': 'GenerativeArtificialIntelligenceandAgentsinResearchandTeaching.pdf', 'pages': '8', 'headings': '2.2 Machine Learning (ML): Algorithms and Neural Networks', 'parent_chunk_index': 30, 'subchunk_index': 0, 'preview': "At the heart of AI's rapid development lies machine learning (ML) , a subfield of AI focused on developing algorithms and statistical models that enable computers to learn from data and solve problems without needing explicit instructions for every scenario (Mahesh 2019). The goal is for systems to ...", 'filename': 'GenerativeArtificialIntelligenceandAgentsinResearchandTeaching.pdf', '_stable_id': 'GenerativeArtificialIntelligenceandAgentsinResearchandTeaching::chunk::30.0::2e0653753d75', 'text': "At the heart of AI's rapid development lies machine learning (ML) , a subfield of AI focused on developing algorithms and statistical models that enable computers to learn from data and solve problems without needing explicit instructions for every scenario (Mahesh 2019). The goal is for systems to independently learn patterns from data, classify new information, and make predictions."}, vector=None, shard_key=None, order_value=None),



# {'source': 'CARMA-Collocation-AwareResourceManagerwithGPUMemoryEstimator.pdf', 'pages': '13', 'headings': 'References', 'parent_chunk_index': 83, 'subchunk_index': 0, 'preview': '- [15] K. R. Jayaram, Vinod Muthusamy, Parijat Dube, Vatche Ishakian, Chen Wang, Benjamin Herta, Scott Boag, Diana Arroyo, Asser Tantawi, Archit Verma, Falk Pollok, and Rania Khalaf. 2019. FfDL: A Flexible Multi-Tenant Deep Learning Platform. In Proceedings of the 20th International Middleware Confe...', 'filename': 'CARMA-Collocation-AwareResourceManagerwithGPUMemoryEstimator.pdf', '_stable_id': 'CARMA-Collocation-AwareResourceManagerwithGPUMemoryEstimator::chunk::83.0::6963e885e270', 'text': "- [15] K. R. Jayaram, Vinod Muthusamy, Parijat Dube, Vatche Ishakian, Chen Wang, Benjamin Herta, Scott Boag, Diana Arroyo, Asser Tantawi, Archit Verma, Falk Pollok, and Rania Khalaf. 2019. FfDL: A Flexible Multi-Tenant Deep Learning Platform. In Proceedings of the 20th International Middleware Conference (Davis, CA, USA) (Middleware '19) . Association for Computing Machinery, New York, NY, USA, 82-95. doi: 10.1145/3361525.3361538"}
# {'source': 'AdvancementsinCropAnalysisthroughDeepLearningandExplainableAI.pdf', 'pages': '92', 'headings': 'REFERENCES', 'parent_chunk_index': 223, 'subchunk_index': 0, 'preview': '- [108] G.  Çınarer,  N.  Erbaş  and  A.  Öcal,  "Rice  classification  and  quality  detection  success  with artificial intelligence technologies," Brazilian Archives of Biology and Technology, vol. 67, p. e24220754, 2024.\n- [109] D. Bhatt, C. Patel, H. Talsania, J. Patel, R. Vaghela, S. Pandya, K...', 'filename': 'AdvancementsinCropAnalysisthroughDeepLearningandExplainableAI.pdf', '_stable_id': 'AdvancementsinCropAnalysisthroughDeepLearningandExplainableAI::chunk::223.0::535cb815e3b5', 'text': '- [108] G.  Çınarer,  N.  Erbaş  and  A.  Öcal,  "Rice  classification  and  quality  detection  success  with artificial intelligence technologies," Brazilian Archives of Biology and Technology, vol. 67, p. e24220754, 2024.\n- [109] D. Bhatt, C. Patel, H. Talsania, J. Patel, R. Vaghela, S. Pandya, K. Modi and H. Ghayvat, "CNN variants for computer vision: History, architecture, application, challenges and future scope," Electronics, vol. 10, p. 2470, 2021.\n- [110] M.  Albahar,  "A  survey  on  deep  learning  and  its  impact  on  agriculture:  challenges  and opportunities," Agriculture, vol. 13, p. 540, 2023.\n- [111] M.  Albahar,  "A  survey  on  deep  learning  and  its  impact  on  agriculture:  challenges  and opportunities," Agriculture, vol. 13, p. 540, 2023.'}