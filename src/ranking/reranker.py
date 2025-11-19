"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""
import os
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

# -------------------------- LLM Cache --------------------------
_LLM_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

def get_llm(model_name: str = "microsoft/DialoGPT-medium"):
    """
    Fetch the cached LLM model to prevent reloading on every query.
    Uses a lightweight model for re-ranking efficiency.
    """
    if model_name not in _LLM_CACHE:
        print(f"Loading LLM model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        _LLM_CACHE[model_name] = (tokenizer, model)
        print(f"LLM model loaded successfully!")
    return _LLM_CACHE[model_name]

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        print(f"Loading cross-encoder model: {model_name}...")
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]

def llm_rerank(query: str, chunks: List[str], top_n: int, model_name: str = "microsoft/DialoGPT-medium", llm_weight: float = 0.7) -> List[Tuple[str, float]]:
    """
    LLM-based re-ranking using prompt-based scoring.
    
    This approach uses a language model to score each chunk based on:
    1. Relevance to the query
    2. Information completeness
    3. Clarity and coherence
    
    Args:
        query: The user's query
        chunks: List of text chunks to re-rank
        top_n: Number of top chunks to return
        model_name: HuggingFace model name for LLM
        llm_weight: Weight for LLM score vs quality score (0-1)
    
    Returns:
        List of (chunk, score) tuples, sorted by score in descending order
    """
    tokenizer, model = get_llm(model_name)
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Create a prompt that asks the LLM to evaluate relevance
        prompt = f"""Question: {query}

Context: {chunk[:500]}  # Truncate to avoid token limits

Rate how well this context answers the question on a scale of 1-10:
"""
        
        # Get LLM response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the score from the response
        score = extract_llm_score(response, chunk)
        
        # Also consider chunk quality
        quality_score = calculate_chunk_quality(chunk)
        
        # Combine LLM score with quality score using configurable weight
        combined_score = (score * llm_weight) + (quality_score * (1 - llm_weight))
        
        scored_chunks.append((chunk, combined_score))
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Sort by score and return top_n
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_n]

def extract_llm_score(response: str, chunk: str) -> float:
    """
    Extract a numeric score from the LLM response.
    Falls back to heuristic scoring if no clear score is found.
    """
    import re
    
    # Look for patterns like "7/10", "score: 8", "rating: 9", etc.
    score_patterns = [
        r'(\d+)/10',
        r'score[:\s]*(\d+)',
        r'rating[:\s]*(\d+)',
        r'(\d+)\s*out\s*of\s*10',
        r'(\d+)\s*points?',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response.lower())
        if match:
            try:
                score = float(match.group(1))
                return min(score / 10.0, 1.0)  # Normalize to 0-1
            except ValueError:
                continue
    
    # If no clear score, use heuristic based on response characteristics
    # Check if response contains positive indicators
    positive_words = ['good', 'excellent', 'perfect', 'clear', 'helpful', 'relevant', 'accurate']
    negative_words = ['poor', 'bad', 'unclear', 'irrelevant', 'incomplete', 'wrong']
    
    response_lower = response.lower()
    positive_count = sum(1 for word in positive_words if word in response_lower)
    negative_count = sum(1 for word in negative_words if word in response_lower)
    
    # Base score on sentiment and length
    base_score = 0.5  # Neutral
    if positive_count > negative_count:
        base_score += 0.2 * min(positive_count, 3)
    elif negative_count > positive_count:
        base_score -= 0.2 * min(negative_count, 3)
    
    # Consider chunk length as a factor
    if len(chunk) > 200:  # Substantial content
        base_score += 0.1
    elif len(chunk) < 50:  # Too short
        base_score -= 0.1
    
    return max(0.0, min(1.0, base_score))

# In reranker.py

def calculate_chunk_quality(chunk: str) -> float:
    """
    Calculate a quality score for a chunk based on database-specific heuristics.
    Higher scores indicate better quality chunks for database/RAG content.
    """
    score = 0.0
    
    # Favor chunks with SQL code examples (highly valuable for database content)
    sql_patterns = [
        'select', 'from', 'where', 'insert', 'update', 'delete', 
        'create', 'alter', 'drop', 'join', 'group by', 'order by'
    ]
    sql_keywords = sum(1 for pattern in sql_patterns if pattern.lower() in chunk.lower())
    if sql_keywords >= 2:  # At least 2 SQL keywords suggest a SQL query
        score += 3.0
    elif sql_keywords >= 1:
        score += 1.5
    
    # Favor chunks with code blocks (SQL queries, schemas, etc.)
    if '```' in chunk:
        score += 2.0
    
    # Favor chunks with relational algebra expressions
    relational_algebra_patterns = ['σ', 'π', '⨝', '∪', '∩', '-', 'ρ', 'γ']
    ra_count = sum(1 for pattern in relational_algebra_patterns if pattern in chunk)
    if ra_count > 0:
        score += ra_count * 1.0
    
    # Favor chunks with database schemas (table definitions)
    schema_patterns = ['(', ')', ';', 'char(', 'numeric(', 'int(', 'varchar(']
    if any(pattern in chunk for pattern in schema_patterns):
        score += 1.5
    
    # Favor chunks with bullet points or numbered lists (better structured explanations)
    lines = chunk.split('\n')
    bullet_points = sum(1 for line in lines if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.', '°')))
    score += min(2.0, bullet_points * 0.5)  # Cap at 2 points
    
    # Favor chunks with database-specific definitions and explanations
    db_terms = [
        'relation', 'attribute', 'tuple', 'schema', 'instance', 'primary key', 
        'foreign key', 'normalization', 'transaction', 'concurrency', 'query',
        'database', 'table', 'record', 'field', 'sql', 'relational algebra',
        'entity', 'relationship', 'integrity', 'constraint', 'index'
    ]
    db_term_count = sum(1 for term in db_terms if term.lower() in chunk.lower())
    score += min(2.0, db_term_count * 0.3)  # Cap at 2 points
    
    # Favor chunks with examples and sample data
    example_indicators = ['example', 'for instance', 'such as', 'e.g.', 'fig.', 'figure']
    if any(indicator in chunk.lower() for indicator in example_indicators):
        score += 1.0
    
    # Favor medium-length chunks (not too short, not too long)
    chunk_length = len(chunk.split())
    if 50 <= chunk_length <= 200:
        score += 1.0
    elif chunk_length > 300:  # Very long chunks might be too verbose
        score -= 0.5
    
    # Favor chunks with proper formatting (headings, code blocks, etc.)
    if any(marker in chunk for marker in ['# ', '## ', '### ', '```']):
        score += 1.0
    
    # Penalty for chunks that are just metadata or page markers
    if any(marker in chunk.lower() for marker in ['page', '--- page', 'chapter', 'section']):
        score -= 1.0
    
    return min(max(score / 5.0, 0.0), 1.0)

def improved_rerank(query: str, chunks: List[str], top_n: int) -> List[Tuple[str, float]]:
    """
    Improved re-ranking that combines multiple signals:
    1. Semantic relevance (from cross-encoder)
    2. Chunk quality (structure, length, etc.)
    3. Diversity (to avoid redundant information)
    """
    if not chunks:
        return []
    
    # 1. Get semantic relevance scores
    cross_encoder = get_cross_encoder()
    pairs = [(query, chunk) for chunk in chunks]
    semantic_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    # Normalize scores to 0-1 range
    max_semantic = max(semantic_scores) if any(semantic_scores) else 1.0
    semantic_scores = [s/max_semantic for s in semantic_scores]
    
    # 2. Calculate quality scores for each chunk
    quality_scores = [calculate_chunk_quality(chunk) for chunk in chunks]
    
    # 3. Calculate diversity penalty
    # First, get embeddings for all chunks
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks, show_progress_bar=False)
    
    # 4. Combine scores with weights
    combined_scores = []
    selected_indices = set()
    
    for _ in range(min(top_n, len(chunks))):
        best_score = -1
        best_idx = -1
        
        for i, (sem_score, qual_score) in enumerate(zip(semantic_scores, quality_scores)):
            if i in selected_indices:
                continue
                
            # Calculate diversity penalty based on similarity to already selected chunks
            diversity_penalty = 0.0
            if selected_indices:
                similarities = [np.dot(chunk_embeddings[i], chunk_embeddings[j]) 
                              for j in selected_indices]
                diversity_penalty = max(similarities) if similarities else 0.0
            
            # Combine scores (higher is better)
            combined = (0.6 * sem_score + 0.3 * qual_score - 0.1 * diversity_penalty)
            
            if combined > best_score:
                best_score = combined
                best_idx = i
        
        if best_idx != -1:
            selected_indices.add(best_idx)
            combined_scores.append((chunks[best_idx], best_score))
    
    return combined_scores


# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(query: str, chunks: List[str], top_n: int) -> List[str]:
    """
    Reranks a list of documents using the cross-encoder model.
    """
    if not chunks:
        return []

    model = get_cross_encoder()

    # Create pairs of [query, chunk] for the model
    pairs = [(query, chunk) for chunk in chunks]

    # Predict the scores
    scores = model.predict(pairs, show_progress_bar=False)

    # Combine chunks with their scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    chunk_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Return just the re-ordered chunks
    return [chunk for chunk, score in chunk_with_scores][0:top_n]


# -------------------------- Reranking Router -----------------------------
# def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> List[str]:
#     """
#     Routes to the appropriate reranker based on the mode in the config.
#     """
#     if mode == "cross_encoder":
#         return rerank_with_cross_encoder(query, chunks, top_n)

#     # We can add other re-ranking strategies in the future to switch between them.

#     # Default is to do nothing (no-op).
#     return chunks

# def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> Tuple[List[str], List[str]]:
#     """
#     Routes to the appropriate reranker and returns both original and re-ranked results.
#     Returns: (original_chunks, reranked_chunks)
#     """
#     if not chunks:
#         return [], []
    
#     # Save original order
#     original_chunks = chunks.copy()
    
#     # Apply re-ranking if a valid mode is specified
#     if mode == "cross_encoder":
#         reranked = rerank_with_cross_encoder(query, chunks, top_n)
#     else:
#         # If no valid mode, just take top_n from original
#         reranked = chunks[:top_n]
    
#     # Return both original and re-ranked results
#     return original_chunks[:top_n], reranked

# def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> Tuple[List[str], List[str]]:
#     """
#     Updated rerank function that uses the improved re-ranking strategy.
#     Returns: (original_chunks, reranked_chunks)
#     """
#     if not chunks:
#         return [], []
    
#     # Save original order
#     original_chunks = chunks.copy()
    
#     if mode == "cross_encoder":
#         # Use the basic cross-encoder
#         reranked = rerank_with_cross_encoder(query, chunks, top_n)
#     elif mode == "improved":
#         # Use our improved re-ranking
#         reranked = [chunk for chunk, _ in improved_rerank(query, chunks, top_n)]
#     else:
#         # Fallback: take top_n from original
#         reranked = chunks[:top_n]
    
#     return original_chunks[:top_n], reranked


def rerank(
    query: str, 
    chunks: List[str], 
    mode: str, 
    top_n: int,
    llm_model: str = "microsoft/DialoGPT-medium",
    llm_weight: float = 0.7
) -> Tuple[List[str], List[str]]:
    """
    Rerank chunks based on the specified mode.
    
    Args:
        query: The search query
        chunks: List of chunks to re-rank (already filtered to desired pool size)
        mode: Reranking mode ('cross_encoder', 'improved', 'llm', etc.)
        top_n: Number of chunks to return after re-ranking
        llm_model: HuggingFace model for LLM re-ranking
        llm_weight: Weight for LLM score vs quality score
        
    Returns:
        Tuple of (original_chunks, reranked_chunks) each containing up to top_n chunks
    """
    if not chunks:
        return [], []
    
    # Save original order (first top_n chunks)
    original_chunks = chunks[:top_n]
    
    if mode == "cross_encoder":
        # Use the basic cross-encoder
        reranked = rerank_with_cross_encoder(query, chunks, top_n)
    elif mode == "improved":
        # Use our improved re-ranking strategy
        reranked = [chunk for chunk, _ in improved_rerank(query, chunks, top_n)]
    elif mode == "llm":
        # Use LLM-based re-ranking with configuration
        reranked = [chunk for chunk, _ in llm_rerank(query, chunks, top_n, llm_model, llm_weight)]
    else:
        # Fallback: take top_n from original
        reranked = chunks[:top_n]
    
    return original_chunks, reranked
