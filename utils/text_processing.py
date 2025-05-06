"""
Functions for text cleaning, emoji removal, and other text processing utilities.
"""
import re
import string
import emoji
from typing import List, Dict, Tuple, Union, Optional

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra spaces, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_emojis(text: str) -> str:
    """
    Remove emojis from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with emojis removed
    """
    if not text:
        return ""
    
    # Remove emojis using the emoji package
    text = emoji.replace_emoji(text, '')
    
    # Remove any leftover emoji-like characters
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    
    # Clean up any double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_emojis(text: str) -> List[str]:
    """
    Extract all emojis from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of emojis found in the text
    """
    if not text:
        return []
    
    return [c for c in text if c in emoji.EMOJI_DATA]

def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of hashtags found in the text
    """
    if not text:
        return []
    
    # Extract hashtags using regex
    hashtags = re.findall(r'#(\w+)', text)
    
    return hashtags

def extract_mentions(text: str) -> List[str]:
    """
    Extract @mentions from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of mentions found in the text
    """
    if not text:
        return []
    
    # Extract mentions using regex
    mentions = re.findall(r'@(\w+)', text)
    
    return mentions

def count_text_features(text: str) -> Dict[str, int]:
    """
    Count various features in the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of feature counts
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'emoji_count': 0,
            'hashtag_count': 0,
            'mention_count': 0,
            'url_count': 0
        }
    
    # Count characters
    char_count = len(text)
    
    # Count words
    word_count = len(text.split())
    
    # Count sentences (approximate)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    sentence_count = max(1, sentence_count)  # Ensure at least 1 sentence
    
    # Count emojis
    emoji_count = len(extract_emojis(text))
    
    # Count hashtags
    hashtag_count = len(extract_hashtags(text))
    
    # Count mentions
    mention_count = len(extract_mentions(text))
    
    # Count URLs
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'emoji_count': emoji_count,
        'hashtag_count': hashtag_count,
        'mention_count': mention_count,
        'url_count': url_count
    }

def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple tokenization of text into words.
    
    Args:
        text: Text to tokenize
        lowercase: Whether to convert text to lowercase
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into tokens
    tokens = text.split()
    
    return tokens 