import pandas as pd
import numpy as np
import requests
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import re
from dataclasses import dataclass
from tqdm import tqdm

# LangChain imports
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import StreamingStdOutCallbackHandler

# Initialize the LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # or your preferred model
    model_kwargs={"temperature": 0.2, "max_tokens": 2000},
    streaming=False
)

# Configuration
RAG_ENDPOINT = "YOUR_RAG_ENDPOINT_URL"  # Replace with your RAG endpoint
COMMON_WORDS_THRESHOLD = 0.8  # Higher threshold - only flag very generic utterances
MIN_UTTERANCE_LENGTH = 5  # Lower threshold - be more permissive

# ============================================
# Helper Functions
# ============================================

def call_rag_endpoint(query: str, top_k: int = 5) -> List[Dict]:
    """Call the RAG REST endpoint to retrieve relevant chunks."""
    try:
        response = requests.post(
            RAG_ENDPOINT,
            json={"query": query, "top_k": top_k},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json().get("chunks", [])
        else:
            print(f"RAG endpoint error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error calling RAG endpoint: {e}")
        return []

def get_common_words() -> set:
    """Get a set of common/stop words to avoid."""
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know'
    }
    return common_words

def calculate_common_word_ratio(utterance: str) -> float:
    """Calculate the ratio of common words in an utterance."""
    common_words = get_common_words()
    words = utterance.lower().split()
    if not words:
        return 0.0
    common_count = sum(1 for word in words if word in common_words)
    return common_count / len(words)

def detect_compound_statements(utterance: str) -> bool:
    """Detect if utterance contains compound statements about different topics."""
    # Look for conjunctions that might indicate multiple topics
    compound_indicators = [
        r'\band\s+also\b',
        r'\bas\s+well\s+as\b',
        r'\badditionally\b',
        r'\bfurthermore\b',
        r'\bmoreover\b',
        r';',  # semicolon often separates different thoughts
        r'\bbut\s+also\b'
    ]
    
    for pattern in compound_indicators:
        if re.search(pattern, utterance.lower()):
            # Check if it's actually discussing different topics, not just elaborating
            sentences = re.split(r'[.!?]+', utterance)
            if len([s for s in sentences if len(s.strip()) > 20]) > 2:
                return True
    
    return False

def rewrite_utterance_if_needed(utterance: str, llm) -> Tuple[str, bool]:
    """Rewrite utterance if it contains compound statements or quality issues."""
    prompt = PromptTemplate(
        input_variables=["utterance"],
        template="""Analyze this utterance and rewrite it ONLY if absolutely necessary:

Utterance: {utterance}

Rules:
1. Only rewrite if it contains multiple unrelated topics
2. Keep variations and elaborations - they add value
3. Preserve the original meaning and intent
4. If it's acceptable as is, return it unchanged

Return JSON format:
{{
    "needs_rewrite": true/false,
    "rewritten_utterance": "...",
    "reason": "..."
}}"""
    )
    
    try:
        response = llm.invoke(prompt.format(utterance=utterance))
        result = json.loads(response.content)
        if result.get("needs_rewrite", False):
            return result.get("rewritten_utterance", utterance), True
        return utterance, False
    except:
        return utterance, False

def compare_utterances_with_rag(utterances_group: pd.DataFrame, llm) -> pd.DataFrame:
    """Compare utterances from different sources using RAG to determine quality."""
    
    # Get RAG context for each utterance
    rag_results = {}
    for idx, row in utterances_group.iterrows():
        chunks = call_rag_endpoint(row['utterance'], top_k=3)
        rag_results[idx] = {
            'utterance': row['utterance'],
            'source': row['source'],
            'rag_chunks': chunks,
            'relevance_score': sum([chunk.get('score', 0) for chunk in chunks]) / max(len(chunks), 1)
        }
    
    # Prepare comparison prompt
    comparisons = "\n".join([
        f"ID: {idx}\nSource: {data['source']}\nUtterance: {data['utterance']}\nRAG Relevance Score: {data['relevance_score']:.2f}\n"
        for idx, data in rag_results.items()
    ])
    
    prompt = PromptTemplate(
        input_variables=["comparisons"],
        template="""Compare these utterances from DIFFERENT sources about the same topic using their RAG relevance scores and quality:

{comparisons}

Evaluation criteria:
1. Specificity and clarity
2. Completeness of information
3. RAG relevance score
4. Practical usefulness

Decision rules:
- If one is CLEARLY better (more specific, complete, higher relevance): mark the lower quality one for DELETE
- If both are good but offer different perspectives: mark BOTH for REVIEW
- If they're nearly identical: keep the one with higher relevance, DELETE the other

Return JSON format:
{{
    "decisions": [
        {{
            "id": <utterance_id>,
            "decision": "ACCEPT/REVIEW/DELETE",
            "reason": "...",
            "quality_score": 1-10
        }}
    ]
}}"""
    )
    
    try:
        response = llm.invoke(prompt.format(comparisons=comparisons))
        result = json.loads(response.content)
        
        decisions = {d["id"]: d for d in result["decisions"]}
        
        for idx in utterances_group.index:
            if idx in decisions:
                decision = decisions[idx]
                utterances_group.at[idx, 'status'] = decision['decision']
                utterances_group.at[idx, 'reason'] = decision.get('reason', '')
                utterances_group.at[idx, 'quality_score'] = decision.get('quality_score', 5)
                
        return utterances_group
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        # Default to REVIEW if comparison fails
        utterances_group['status'] = 'REVIEW'
        utterances_group['reason'] = 'Comparison failed - needs human review'
        return utterances_group

def handle_same_source_variations(utterances_group: pd.DataFrame) -> pd.DataFrame:
    """Handle utterances from the same source - keep variations as they add value."""
    
    # Check if utterances are too similar (near duplicates)
    utterances_list = utterances_group['utterance'].tolist()
    
    for i, row1 in enumerate(utterances_group.itertuples()):
        for j, row2 in enumerate(utterances_group.itertuples()):
            if i >= j:
                continue
                
            # Simple similarity check (you could use more sophisticated methods)
            utt1 = row1.utterance.lower().strip()
            utt2 = row2.utterance.lower().strip()
            
            # If they're nearly identical (>90% similar)
            if utt1 == utt2 or (len(utt1) > 0 and len(set(utt1.split()) & set(utt2.split())) / len(set(utt1.split()) | set(utt2.split())) > 0.9):
                # Mark the shorter one for deletion
                if len(utt1) < len(utt2):
                    utterances_group.at[row1.Index, 'status'] = 'DELETE'
                    utterances_group.at[row1.Index, 'reason'] = 'Near duplicate from same source'
                else:
                    utterances_group.at[row2.Index, 'status'] = 'DELETE'
                    utterances_group.at[row2.Index, 'reason'] = 'Near duplicate from same source'
            else:
                # They're variations - keep both
                utterances_group.at[row1.Index, 'status'] = 'ACCEPT'
                utterances_group.at[row1.Index, 'reason'] = 'Valuable variation from same source'
                utterances_group.at[row2.Index, 'status'] = 'ACCEPT'
                utterances_group.at[row2.Index, 'reason'] = 'Valuable variation from same source'
    
    return utterances_group

# ============================================
# Main Processing Pipeline
# ============================================

def process_utterances(df: pd.DataFrame, source_context: Dict[str, str] = None) -> pd.DataFrame:
    """Main pipeline to process utterances with conservative deletion approach."""
    
    # Add new columns
    df['status'] = 'ACCEPT'  # Default status - keep by default
    df['modified_utterance'] = ''
    df['reason'] = ''
    df['quality_score'] = 5  # Default middle score
    df['common_word_ratio'] = df['utterance'].apply(calculate_common_word_ratio)
    df['has_compound'] = df['utterance'].apply(detect_compound_statements)
    
    print("Step 1: Grouping utterances by topic...")
    
    # Group by topic (title + category combination)
    topic_groups = defaultdict(list)
    for idx, row in df.iterrows():
        topic_key = f"{row['category']}_{row['title']}"
        topic_groups[topic_key].append(idx)
    
    # Separate groups by source diversity
    same_source_groups = {}
    multi_source_groups = {}
    
    for topic, indices in topic_groups.items():
        if len(indices) > 1:
            group_df = df.loc[indices]
            unique_sources = group_df['source'].nunique()
            
            if unique_sources == 1:
                same_source_groups[topic] = indices
            else:
                multi_source_groups[topic] = indices
    
    print(f"Found {len(same_source_groups)} topics with multiple utterances from same source")
    print(f"Found {len(multi_source_groups)} topics with utterances from different sources")
    
    print("\nStep 2: Processing same-source variations (keeping valuable variations)...")
    
    for topic, indices in tqdm(same_source_groups.items(), desc="Same source groups"):
        group_df = df.loc[indices].copy()
        
        # Handle same source variations - generally keep them
        updated_group = handle_same_source_variations(group_df)
        
        # Update main dataframe
        for idx, row in updated_group.iterrows():
            df.at[idx, 'status'] = row['status']
            df.at[idx, 'reason'] = row.get('reason', '')
    
    print("\nStep 3: Processing multi-source groups (comparing quality)...")
    
    for topic, indices in tqdm(multi_source_groups.items(), desc="Multi-source groups"):
        group_df = df.loc[indices].copy()
        
        # Group by source within this topic
        source_groups = group_df.groupby('source')
        
        if len(source_groups) > 1:
            # Compare utterances from different sources
            compared_group = compare_utterances_with_rag(group_df, llm)
            
            # Update main dataframe
            for idx, row in compared_group.iterrows():
                df.at[idx, 'status'] = row['status']
                df.at[idx, 'reason'] = row.get('reason', '')
                df.at[idx, 'quality_score'] = row.get('quality_score', 5)
    
    print("\nStep 4: Checking for quality issues (but being conservative)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Quality check"):
        # Only suggest modifications for severe issues
        
        # Very high common word ratio (only if extremely generic)
        if row['common_word_ratio'] > 0.9 and row['status'] == 'ACCEPT':
            df.at[idx, 'status'] = 'MODIFY'
            df.at[idx, 'reason'] = 'Extremely generic - needs more specificity'
            
            # Try to rewrite
            rewritten, needed = rewrite_utterance_if_needed(row['utterance'], llm)
            if needed:
                df.at[idx, 'modified_utterance'] = rewritten
        
        # Severe compound statements
        elif row['has_compound'] and row['status'] == 'ACCEPT':
            # Double-check if it really needs modification
            rewritten, needed = rewrite_utterance_if_needed(row['utterance'], llm)
            if needed:
                df.at[idx, 'status'] = 'MODIFY'
                df.at[idx, 'reason'] = 'Contains unrelated compound statements'
                df.at[idx, 'modified_utterance'] = rewritten
        
        # Only delete if extremely short AND meaningless
        elif len(row['utterance'].strip()) < MIN_UTTERANCE_LENGTH:
            # Check if it's actually meaningless
            if row['common_word_ratio'] > 0.95:
                df.at[idx, 'status'] = 'DELETE'
                df.at[idx, 'reason'] = 'Too short and contains no meaningful content'
            else:
                df.at[idx, 'status'] = 'REVIEW'
                df.at[idx, 'reason'] = 'Very short - needs human review'
    
    print("\nStep 5: Final validation of DELETE decisions...")
    
    # Review all DELETE decisions to ensure they're justified
    delete_rows = df[df['status'] == 'DELETE']
    if len(delete_rows) > 0:
        print(f"Validating {len(delete_rows)} deletion decisions...")
        
        for idx, row in delete_rows.iterrows():
            # Ensure deletion has strong justification
            if 'duplicate' not in row['reason'].lower() and 'quality' not in row['reason'].lower():
                # Reconsider - change to REVIEW instead
                df.at[idx, 'status'] = 'REVIEW'
                df.at[idx, 'reason'] = f"Originally marked for deletion: {row['reason']} - needs human review"
    
    return df

def generate_detailed_report(df: pd.DataFrame) -> Dict:
    """Generate a detailed report of the processing results."""
    
    status_counts = df['status'].value_counts().to_dict()
    
    report = {
        'summary': {
            'total_utterances': len(df),
            'accepted': status_counts.get('ACCEPT', 0),
            'marked_for_review': status_counts.get('REVIEW', 0),
            'marked_for_modification': status_counts.get('MODIFY', 0),
            'marked_for_deletion': status_counts.get('DELETE', 0),
            'deletion_percentage': f"{(status_counts.get('DELETE', 0) / len(df) * 100):.1f}%"
        },
        'by_source': {},
        'by_category': {},
        'deletion_reasons': [],
        'modification_reasons': []
    }
    
    # Source breakdown
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        report['by_source'][source] = {
            'total': len(source_df),
            'status_breakdown': source_df['status'].value_counts().to_dict()
        }
    
    # Category breakdown
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        report['by_category'][category] = {
            'total': len(cat_df),
            'status_breakdown': cat_df['status'].value_counts().to_dict()
        }
    
    # Deletion reasons
    delete_df = df[df['status'] == 'DELETE']
    if len(delete_df) > 0:
        report['deletion_reasons'] = delete_df[['utterance', 'source', 'reason']].to_dict('records')
    
    # Modification reasons
    modify_df = df[df['status'] == 'MODIFY']
    if len(modify_df) > 0:
        report['modification_reasons'] = modify_df[['utterance', 'modified_utterance', 'reason']].to_dict('records')
    
    return report

# ============================================
# Main Execution
# ============================================

# Load your data
# df = pd.read_csv('your_utterances.csv')  # Replace with your file path

# Example usage with sample data
sample_data = {
    'utterance': [
        'How do I reset my password?',
        'Steps to reset your password in our system',
        'Password reset process explained in detail',
        'I need to reset my password quickly',
        'The weather is nice and I also need help with login',
        'Tell me about account security',
        'Account security best practices and recommendations',
        'Security measures for your account',
        'How to secure your account properly'
    ],
    'source': ['source1', 'source1', 'source2', 'source3', 'source1', 'source2', 'source2', 'source3', 'source3'],
    'title': ['password_reset', 'password_reset', 'password_reset', 'password_reset', 'misc', 'security', 'security', 'security', 'security'],
    'category': ['account', 'account', 'account', 'account', 'general', 'account', 'account', 'account', 'account']
}

df = pd.DataFrame(sample_data)

# Optional: Add source context if you have it
source_context = {
    'source1': 'Internal documentation and help center',
    'source2': 'Customer support knowledge base',
    'source3': 'Community forums and FAQs'
}

# Process the utterances
print("Starting conservative utterance processing pipeline...")
print("=" * 60)
processed_df = process_utterances(df, source_context)

# Generate detailed report
report = generate_detailed_report(processed_df)

print("\n" + "="*60)
print("PROCESSING REPORT")
print("="*60)
print("\nSUMMARY:")
for key, value in report['summary'].items():
    print(f"  {key}: {value}")

print("\nBY SOURCE:")
for source, data in report['by_source'].items():
    print(f"  {source}:")
    print(f"    Total: {data['total']}")
    for status, count in data['status_breakdown'].items():
        print(f"    {status}: {count}")

if report['deletion_reasons']:
    print("\nDELETION DECISIONS (requiring strong justification):")
    for item in report['deletion_reasons'][:5]:  # Show first 5
        print(f"  - '{item['utterance'][:50]}...' ({item['source']})")
        print(f"    Reason: {item['reason']}")

# Save results
processed_df.to_csv('processed_utterances.csv', index=False)
print("\nResults saved to 'processed_utterances.csv'")

# Display sample of results
print("\n" + "="*60)
print("SAMPLE OF PROCESSED DATA:")
print("="*60)
display_columns = ['utterance', 'source', 'status', 'reason']
print(processed_df[display_columns].head(10).to_string())

# Show statistics
print("\n" + "="*60)
print("FINAL STATISTICS:")
print("="*60)
print(f"Utterances kept as-is: {len(processed_df[processed_df['status'] == 'ACCEPT'])}")
print(f"Utterances for review: {len(processed_df[processed_df['status'] == 'REVIEW'])}")
print(f"Utterances to modify: {len(processed_df[processed_df['status'] == 'MODIFY'])}")
print(f"Utterances to delete: {len(processed_df[processed_df['status'] == 'DELETE'])}")
print(f"Deletion rate: {(len(processed_df[processed_df['status'] == 'DELETE']) / len(processed_df) * 100):.1f}%")