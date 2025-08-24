#!/usr/bin/env python3

import fitz
import re
import json
import statistics
import os
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import math
import unicodedata
from pathlib import Path
import langid

class ak_97_DocumentStructureAnalyzer:
    
    def __init__(self):
        self.ak_97_settings = {
            "TITLE_SIZE_MULTIPLIER": 1.05,
            "MARGIN_BOUNDARY": 0.10,
            "MAX_SECTION_LEVELS": 4,
            "MIN_CONTENT_LENGTH_LATIN": 5,
            "MIN_CONTENT_LENGTH_CJK": 5,
            "MIN_CONTENT_LENGTH_INDIC": 5,
            "MIN_CONTENT_LENGTH_ARABIC": 5,
            "MIN_CONTENT_LENGTH_CYRILLIC": 5,
            "MAX_SECTION_WORDS": 30,
            "MAX_DOCUMENT_TITLE_WORDS": 35,
        }
        
        self.ak_97_character_based_languages = {
            "zh", "ja", "ko", "th", "my", "km", "lo", "bo", "dz"
        }
        
        self.ak_97_indic_languages = {
            "hi", "bn", "mr", "gu", "ta", "te", "kn", "ml", "or", "pa", "as", "ne",
            "si", "ur", "sd", "ks", "sa", "pi", "bh", "mai", "bho", "new", "gom"
        }
        
        self.ak_97_arabic_script_languages = {
            "ar", "fa", "ps", "ur", "ug", "ku", "sd", "ks", "bal", "ckb"
        }
        
        self.ak_97_cyrillic_languages = {
            "ru", "uk", "bg", "sr", "mk", "be", "mn", "kk", "ky", "tg", "uz",
            "ce", "cv", "tt", "ba", "sah", "myv", "mdf", "kv", "udm", "koi"
        }
        
        self.ak_97__prepare_filter_patterns()
    
    def ak_97__prepare_filter_patterns(self):

        self.ak_97_content_filters = [
            re.compile(r'(?i)copyright\s*[©®™]?'),
            re.compile(r'(?i)©\s*\d{4}'),
            re.compile(r'(?i)version\s*\d+'),
            re.compile(r'(?i)all\s+rights\s+reserved'),
            re.compile(r'(?i)www\.\w+\.\w+'),
            re.compile(r'(?i)http[s]?://\S+'),
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        ]
        
        self.ak_97_field_label_filter = re.compile(r'^\d+\.\s*$|^\w+\.\s*$|^[A-Z][a-z]*\s+of\s+\w+\s*[.:]?\s*$')
    
    def ak_97_identify_language(self, ak_97_input_text: str) -> str:

        if not ak_97_input_text or not ak_97_input_text.strip():
            return "en"
        try:
            ak_97_detected_lang, _ = langid.classify(ak_97_input_text)
            return ak_97_detected_lang
        except:
            return "en"
    
    def ak_97_retrieve_min_content_threshold(self, ak_97_lang_code: str) -> int:

        if ak_97_lang_code in self.ak_97_character_based_languages:
            return self.ak_97_settings["MIN_CONTENT_LENGTH_CJK"]
        elif ak_97_lang_code in self.ak_97_indic_languages:
            return self.ak_97_settings["MIN_CONTENT_LENGTH_INDIC"]
        elif ak_97_lang_code in self.ak_97_arabic_script_languages:
            return self.ak_97_settings["MIN_CONTENT_LENGTH_ARABIC"]
        elif ak_97_lang_code in self.ak_97_cyrillic_languages:
            return self.ak_97_settings["MIN_CONTENT_LENGTH_CYRILLIC"]
        else:
            return self.ak_97_settings["MIN_CONTENT_LENGTH_LATIN"]
    
    def ak_97_process_document_structure(self, ak_97_file_path: str) -> Tuple[str, List[Dict], str]:

        try:
            ak_97_file_document = fitz.open(ak_97_file_path)
        except Exception as e:
            return "", [], ""
        
        # Get preview text for language detection
        ak_97_preview_text = ""
        for i in range(min(3, len(ak_97_file_document))):
            ak_97_preview_text += ak_97_file_document[i].get_text()
        
        ak_97_detected_language = self.ak_97_identify_language(ak_97_preview_text)
        ak_97_document_title = self.ak_97__derive_document_title(ak_97_file_document)
        
        # Check for built-in TOC first
        ak_97_built_in_contents = ak_97_file_document.get_toc(simple=True)
        if ak_97_built_in_contents:
            ak_97_structure_outline = self.ak_97__handle_built_in_contents(ak_97_built_in_contents)
        else:
            ak_97_structure_outline = self.ak_97__build_structure_from_analysis(
                ak_97_file_document, document_title=ak_97_document_title, lang_code=ak_97_detected_language
            )
        
        # Get full text
        ak_97_full_text = ""
        try:
            for ak_97_page_num in range(len(ak_97_file_document)):
                ak_97_page = ak_97_file_document[ak_97_page_num]
                ak_97_full_text += ak_97_page.get_text() + "\n"
        except Exception:
            pass
        
        ak_97_file_document.close()
        return ak_97_document_title, ak_97_structure_outline, ak_97_full_text
    
    def ak_97__derive_document_title(self, ak_97_file_document: fitz.Document) -> str:

        try:
            ak_97_opening_page = ak_97_file_document[0]
            ak_97_content_blocks = ak_97_opening_page.get_text("dict", flags=4)["blocks"]
            ak_97_extracted_lines = []
            
            for ak_97_block in ak_97_content_blocks:
                for ak_97_text_line in ak_97_block.get("lines", []):
                    ak_97_text_spans = ak_97_text_line.get("spans", [])
                    if not ak_97_text_spans:
                        continue
                    
                    ak_97_combined_text = " ".join(ak_97_text_span["text"] for ak_97_text_span in ak_97_text_spans).strip()
                    if ak_97_combined_text:
                        ak_97_extracted_lines.append({
                            "text": ak_97_combined_text,
                            "size": ak_97_text_spans[0]["size"],
                            "y0": ak_97_text_line["bbox"][1],
                            "y1": ak_97_text_line["bbox"][3],
                        })
            
            if not ak_97_extracted_lines:
                return ""
            
            # Find largest text in top third of page
            ak_97_page_height = ak_97_opening_page.rect.height
            ak_97_top_third_lines = [ak_97_line for ak_97_line in ak_97_extracted_lines if ak_97_line["y0"] < ak_97_page_height / 3]
            
            if not ak_97_top_third_lines:
                return ""
            
            ak_97_largest_size = max(ak_97_line["size"] for ak_97_line in ak_97_top_third_lines)
            ak_97_title_lines = [ak_97_line for ak_97_line in ak_97_top_third_lines if ak_97_line["size"] >= ak_97_largest_size * 0.9]
            ak_97_title_lines.sort(key=lambda x: x["y0"])
            
            ak_97_title = " ".join(ak_97_line["text"] for ak_97_line in ak_97_title_lines).strip()
            return ak_97_title[:200] if len(ak_97_title) > 200 else ak_97_title
            
        except Exception:
            return ""
    
    def ak_97__handle_built_in_contents(self, ak_97_built_in_contents: List) -> List[Dict]:

        if not ak_97_built_in_contents:
            return []
        
        ak_97_hierarchy_levels = [ak_97_item[0] for ak_97_item in ak_97_built_in_contents]
        ak_97_base_level = min(ak_97_hierarchy_levels) if ak_97_hierarchy_levels else 1
        
        return [
            {
                "level": f"H{ak_97_item[0] - ak_97_base_level + 1}",
                "text": ak_97_item[1].strip(),
                "page": ak_97_item[2] - 1
            }
            for ak_97_item in ak_97_built_in_contents
            if ak_97_item[2] > 0
        ]
    
    def ak_97__build_structure_from_analysis(self, ak_97_file_document: fitz.Document,
                                           document_title: str = "", lang_code: str = "en") -> List[Dict]:

        ak_97_extracted_text_lines = self.ak_97__gather_all_text_lines(ak_97_file_document)
        if not ak_97_extracted_text_lines:
            return []
        
        ak_97_standard_text_size = self.ak_97__establish_standard_text_size(ak_97_extracted_text_lines, lang_code)
        ak_97_section_candidates = self.ak_97__discover_section_candidates(
            ak_97_extracted_text_lines, ak_97_standard_text_size, document_title, lang_code
        )
        
        return self.ak_97__organize_and_combine_sections(ak_97_section_candidates)
    
    def ak_97__gather_all_text_lines(self, ak_97_file_document: fitz.Document) -> List[Dict]:

        ak_97_collected_lines = []
        
        for ak_97_page_index in range(len(ak_97_file_document)):
            ak_97_document_page = ak_97_file_document[ak_97_page_index]
            ak_97_content_blocks = ak_97_document_page.get_text("dict", flags=0)["blocks"]
            
            for ak_97_block in ak_97_content_blocks:
                for ak_97_text_line in ak_97_block.get("lines", []):
                    ak_97_text_spans = ak_97_text_line.get("spans", [])
                    if not ak_97_text_spans:
                        continue
                    
                    ak_97_combined_text = " ".join(ak_97_text_span["text"] for ak_97_text_span in ak_97_text_spans).strip()
                    if not ak_97_combined_text:
                        continue
                    
                    # Calculate weighted font size
                    ak_97_total_span_length = sum(len(ak_97_text_span["text"]) for ak_97_text_span in ak_97_text_spans)
                    if ak_97_total_span_length == 0:
                        ak_97_weighted_size = 0
                    else:
                        ak_97_weighted_size = sum(
                            ak_97_text_span["size"] * len(ak_97_text_span["text"]) for ak_97_text_span in ak_97_text_spans
                        ) / ak_97_total_span_length
                    
                    ak_97_collected_lines.append({
                        "text": ak_97_combined_text,
                        "size": ak_97_weighted_size,
                        "bbox": ak_97_text_line["bbox"],
                        "page_num": ak_97_page_index,
                        "spans": ak_97_text_spans,
                    })
        
        return ak_97_collected_lines
    
    def ak_97__establish_standard_text_size(self, ak_97_collected_lines: List[Dict], ak_97_lang_code: str) -> int:

        ak_97_threshold_length = self.ak_97_retrieve_min_content_threshold(ak_97_lang_code)
        ak_97_main_text_candidates = [ak_97_line for ak_97_line in ak_97_collected_lines if len(ak_97_line["text"]) > ak_97_threshold_length]
        
        if not ak_97_main_text_candidates:
            ak_97_main_text_candidates = ak_97_collected_lines
        
        ak_97_size_distribution = Counter(round(ak_97_line["size"]) for ak_97_line in ak_97_main_text_candidates)
        return ak_97_size_distribution.most_common(1)[0][0] if ak_97_size_distribution else 12
    
    def ak_97__discover_section_candidates(self, ak_97_collected_lines: List[Dict], ak_97_standard_text_size: int,
                                          ak_97_document_title: str, ak_97_lang_code: str) -> List[Dict]:

        ak_97_section_candidates = []
        
        for ak_97_text_line in ak_97_collected_lines:
            if self.ak_97__is_potential_heading(ak_97_text_line, ak_97_standard_text_size, ak_97_document_title, ak_97_lang_code):
                ak_97_section_candidates.append(ak_97_text_line)
        
        return ak_97_section_candidates
    
    def ak_97__is_potential_heading(self, ak_97_text_line: Dict, ak_97_standard_text_size: int,
                                   ak_97_document_title: str, ak_97_lang_code: str) -> bool:

        ak_97_text = ak_97_text_line["text"].strip()
        
        # Basic validation
        if len(ak_97_text) < 2 or len(ak_97_text) > 200:
            return False
        
        # Filter out metadata patterns
        if any(ak_97_pattern.search(ak_97_text) for ak_97_pattern in self.ak_97_content_filters):
            return False
        
        # Check if it matches document title (exclude)
        if ak_97_document_title and ak_97_text.lower() in ak_97_document_title.lower():
            return False
        
        # Font size check
        ak_97_size_threshold = ak_97_standard_text_size * self.ak_97_settings["TITLE_SIZE_MULTIPLIER"]
        if ak_97_text_line["size"] >= ak_97_size_threshold:
            return True
        
        # Bold text check
        if ak_97_text_line.get("spans"):
            ak_97_first_span = ak_97_text_line["spans"][0]
            ak_97_font_name = ak_97_first_span.get("font", "").lower()
            ak_97_is_bold = "bold" in ak_97_font_name or (ak_97_first_span.get("flags", 0) & 16) != 0
            
            if ak_97_is_bold and len(ak_97_text.split()) <= 15:
                return True
        
        # Numbered section check
        if re.match(r'^(\d+\.|\d+\.\d+|\w+\.)\s+', ak_97_text):
            return True
        
        return False
    
    def ak_97__organize_and_combine_sections(self, ak_97_combined_sections: List[Dict]) -> List[Dict]:

        if not ak_97_combined_sections:
            return []
        
        # Group by font size to determine hierarchy
        ak_97_size_groups = defaultdict(list)
        for ak_97_section in ak_97_combined_sections:
            ak_97_size_groups[round(ak_97_section["size"])].append(ak_97_section)
        
        # Sort sizes descending (larger = higher level)
        ak_97_sorted_sizes = sorted(ak_97_size_groups.keys(), reverse=True)
        
        # Assign levels (max 4 levels as per settings)
        ak_97_max_levels = min(len(ak_97_sorted_sizes), self.ak_97_settings["MAX_SECTION_LEVELS"])
        ak_97_size_to_level = {ak_97_size: f"H{i+1}" for i, ak_97_size in enumerate(ak_97_sorted_sizes[:ak_97_max_levels])}
        
        ak_97_structured_sections = []
        for ak_97_section in ak_97_combined_sections:
            ak_97_size = round(ak_97_section["size"])
            ak_97_level = ak_97_size_to_level.get(ak_97_size)
            if ak_97_level:
                ak_97_structured_sections.append({
                    "level": ak_97_level,
                    "text": ak_97_section["text"],
                    "page": ak_97_section["page_num"]
                })
        
        # Sort by page and position
        ak_97_structured_sections.sort(key=lambda x: (x["page"], x.get("bbox", [0, 0, 0, 0])[1]))
        return ak_97_structured_sections

class ak_97_SmallLanguageModel:
    
    def __init__(self):
        self.ak_97_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
    
    def ak_97_analyze_content_relevance(self, ak_97_text: str, ak_97_persona: str, ak_97_task: str) -> float:

        ak_97_text_lower = ak_97_text.lower()
        
        # Extract meaningful keywords dynamically from persona and task
        ak_97_query_keywords = self.ak_97__extract_meaningful_keywords(f"{ak_97_persona} {ak_97_task}")
        ak_97_text_keywords = self.ak_97__extract_meaningful_keywords(ak_97_text_lower)
        
        if not ak_97_query_keywords:
            return 0.0
        
        # Calculate different types of relevance scores
        ak_97_scores = []
        
        # 1. Direct keyword overlap
        ak_97_direct_overlap = len(ak_97_query_keywords & ak_97_text_keywords)
        ak_97_direct_score = ak_97_direct_overlap / len(ak_97_query_keywords)
        ak_97_scores.append(ak_97_direct_score * 0.4)
        
        # 2. Semantic similarity using n-grams
        ak_97_semantic_score = self.ak_97__calculate_ngram_similarity(ak_97_text_lower, ak_97_persona.lower(), ak_97_task.lower())
        ak_97_scores.append(ak_97_semantic_score * 0.3)
        
        # 3. Contextual patterns
        ak_97_context_score = self.ak_97__calculate_contextual_relevance(ak_97_text_lower, ak_97_persona.lower(), ak_97_task.lower())
        ak_97_scores.append(ak_97_context_score * 0.2)
        
        # 4. Term frequency importance
        ak_97_tf_score = self.ak_97__calculate_term_frequency_score(ak_97_text_keywords, ak_97_query_keywords)
        ak_97_scores.append(ak_97_tf_score * 0.1)
        
        return min(sum(ak_97_scores), 1.0)
    
    def ak_97__extract_meaningful_keywords(self, ak_97_text: str) -> set:

        # Extract words that are likely to be meaningful
        ak_97_words = re.findall(r'\b[a-zA-Z]{3,}\b', ak_97_text.lower())
        
        # Filter out stopwords and very common words
        ak_97_meaningful_words = set()
        for ak_97_word in ak_97_words:
            if (ak_97_word not in self.ak_97_stopwords and
                len(ak_97_word) >= 3 and
                not ak_97_word.isdigit() and
                not self.ak_97__is_common_function_word(ak_97_word)):
                ak_97_meaningful_words.add(ak_97_word)
        
        return ak_97_meaningful_words
    
    def ak_97__is_common_function_word(self, ak_97_word: str) -> bool:

        ak_97_common_function_words = {
            'will', 'would', 'could', 'should', 'might', 'must', 'can',
            'also', 'just', 'only', 'even', 'still', 'yet', 'already',
            'very', 'quite', 'rather', 'really', 'much', 'many', 'more',
            'most', 'some', 'any', 'each', 'every', 'all', 'both',
            'such', 'same', 'other', 'another', 'different', 'various',
            'several', 'few', 'little', 'less', 'least', 'best', 'better',
            'good', 'great', 'small', 'large', 'big', 'long', 'short',
            'high', 'low', 'new', 'old', 'first', 'last', 'next', 'previous'
        }
        
        return ak_97_word in ak_97_common_function_words
    
    def ak_97__calculate_ngram_similarity(self, ak_97_text: str, ak_97_persona: str, ak_97_task: str) -> float:

        ak_97_query_text = f"{ak_97_persona} {ak_97_task}"
        
        # Generate trigrams for both texts
        ak_97_text_trigrams = self.ak_97__generate_ngrams(ak_97_text, 3)
        ak_97_query_trigrams = self.ak_97__generate_ngrams(ak_97_query_text, 3)
        
        if not ak_97_query_trigrams:
            return 0.0
        
        # Calculate Jaccard similarity
        ak_97_intersection = len(ak_97_text_trigrams & ak_97_query_trigrams)
        ak_97_union = len(ak_97_text_trigrams | ak_97_query_trigrams)
        return ak_97_intersection / ak_97_union if ak_97_union > 0 else 0.0
    
    def ak_97__generate_ngrams(self, ak_97_text: str, ak_97_n: int) -> set:

        # Clean text and create n-grams
        ak_97_clean_text = re.sub(r'[^a-zA-Z\s]', '', ak_97_text.lower())
        ak_97_words = ak_97_clean_text.split()
        ak_97_ngrams = set()
        
        for ak_97_word in ak_97_words:
            if len(ak_97_word) >= ak_97_n:
                for i in range(len(ak_97_word) - ak_97_n + 1):
                    ak_97_ngrams.add(ak_97_word[i:i+ak_97_n])
        
        return ak_97_ngrams
    
    def ak_97__calculate_contextual_relevance(self, ak_97_text: str, ak_97_persona: str, ak_97_task: str) -> float:

        ak_97_score = 0.0
        
        # Look for action verbs from task in text
        ak_97_task_verbs = self.ak_97__extract_action_verbs(ak_97_task)
        ak_97_text_verbs = self.ak_97__extract_action_verbs(ak_97_text)
        ak_97_verb_overlap = len(ak_97_task_verbs & ak_97_text_verbs)
        
        if ak_97_task_verbs:
            ak_97_score += (ak_97_verb_overlap / len(ak_97_task_verbs)) * 0.5
        
        # Look for noun phrases similarity
        ak_97_persona_nouns = self.ak_97__extract_noun_phrases(ak_97_persona)
        ak_97_text_nouns = self.ak_97__extract_noun_phrases(ak_97_text)
        ak_97_noun_overlap = len(ak_97_persona_nouns & ak_97_text_nouns)
        
        if ak_97_persona_nouns:
            ak_97_score += (ak_97_noun_overlap / len(ak_97_persona_nouns)) * 0.3
        
        # Check for numerical/quantitative patterns if present in task
        if self.ak_97__has_quantitative_terms(ak_97_task) and self.ak_97__has_quantitative_terms(ak_97_text):
            ak_97_score += 0.2
        
        return min(ak_97_score, 1.0)
    
    def ak_97__extract_action_verbs(self, ak_97_text: str) -> set:

        # Simple heuristic: words ending in common verb patterns
        ak_97_words = re.findall(r'\b[a-zA-Z]+\b', ak_97_text.lower())
        ak_97_verbs = set()
        ak_97_verb_patterns = ['ing', 'ed', 'ize', 'ise', 'ate', 'fy']
        ak_97_action_words = ['plan', 'create', 'manage', 'analyze', 'prepare', 'develop',
                             'design', 'organize', 'review', 'study', 'research', 'build',
                             'implement', 'execute', 'coordinate', 'facilitate', 'optimize']
        
        for ak_97_word in ak_97_words:
            if (ak_97_word in ak_97_action_words or
                any(ak_97_word.endswith(ak_97_pattern) for ak_97_pattern in ak_97_verb_patterns) or
                ak_97_word.endswith('e') and len(ak_97_word) > 4):  # Simple verb heuristic
                ak_97_verbs.add(ak_97_word)
        
        return ak_97_verbs
    
    def ak_97__extract_noun_phrases(self, ak_97_text: str) -> set:

        ak_97_words = re.findall(r'\b[a-zA-Z]+\b', ak_97_text.lower())
        ak_97_nouns = set()
        
        # Look for words that are likely nouns (simple heuristics)
        for ak_97_word in ak_97_words:
            if (len(ak_97_word) > 3 and
                ak_97_word not in self.ak_97_stopwords and
                not ak_97_word.endswith('ing') and
                not ak_97_word.endswith('ly')):
                ak_97_nouns.add(ak_97_word)
        
        return ak_97_nouns
    
    def ak_97__has_quantitative_terms(self, ak_97_text: str) -> bool:

        ak_97_quantitative_patterns = [
            r'\b\d+\b',  # numbers
            r'\b(few|many|several|multiple|various)\b',
            r'\b(day|week|month|year|hour|minute)s?\b',
            r'\b(first|second|third|last|final)\b'
        ]
        
        return any(re.search(ak_97_pattern, ak_97_text.lower()) for ak_97_pattern in ak_97_quantitative_patterns)
    
    def ak_97__calculate_term_frequency_score(self, ak_97_text_keywords: set, ak_97_query_keywords: set) -> float:

        if not ak_97_query_keywords:
            return 0.0
        
        # Simple TF scoring - count of matching important terms
        ak_97_important_matches = ak_97_text_keywords & ak_97_query_keywords
        
        # Weight longer words more heavily (they're likely more specific)
        ak_97_weighted_score = sum(len(ak_97_word) / 10 for ak_97_word in ak_97_important_matches)
        ak_97_max_possible_score = sum(len(ak_97_word) / 10 for ak_97_word in ak_97_query_keywords)
        
        return ak_97_weighted_score / ak_97_max_possible_score if ak_97_max_possible_score > 0 else 0.0
    
    def ak_97_extract_key_concepts(self, ak_97_text: str, ak_97_max_concepts: int = 10) -> List[str]:

        ak_97_keywords = self.ak_97__extract_meaningful_keywords(ak_97_text)
        
        # Score keywords by length and rarity (longer = more specific)
        ak_97_scored_keywords = [(ak_97_word, len(ak_97_word)) for ak_97_word in ak_97_keywords]
        ak_97_scored_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return [ak_97_word for ak_97_word, _ in ak_97_scored_keywords[:ak_97_max_concepts]]

class ak_97_PersonaDrivenAnalyzer:
    
    def __init__(self):
        self.ak_97_structure_analyzer = ak_97_DocumentStructureAnalyzer()
        self.ak_97_slm = ak_97_SmallLanguageModel()
    
    def ak_97_process_collection(self, ak_97_collection_path: str) -> Dict[str, Any]:

        ak_97_input_config_path = os.path.join(ak_97_collection_path, "challenge1b_input.json")
        
        if not os.path.exists(ak_97_input_config_path):
            raise FileNotFoundError(f"Input configuration file not found: {ak_97_input_config_path}")
        
        with open(ak_97_input_config_path, 'r', encoding='utf-8') as f:
            ak_97_input_config = json.load(f)
        
        # Extract configuration
        ak_97_documents = ak_97_input_config.get("documents", [])
        ak_97_persona_role = ak_97_input_config.get("persona", {}).get("role", "")
        ak_97_job_task = ak_97_input_config.get("job_to_be_done", {}).get("task", "")
        
        print(f"Processing {len(ak_97_documents)} documents")
        print(f"Persona: {ak_97_persona_role}")
        print(f"Task: {ak_97_job_task}")
        
        # Process documents
        ak_97_all_sections = self.ak_97__process_documents(ak_97_collection_path, ak_97_documents, ak_97_persona_role, ak_97_job_task)
        
        # Rank sections by relevance
        ak_97_ranked_sections = self.ak_97__rank_sections_by_relevance(ak_97_all_sections, ak_97_persona_role, ak_97_job_task)
        
        # Generate output
        return self.ak_97__generate_output(ak_97_documents, ak_97_persona_role, ak_97_job_task, ak_97_ranked_sections)
    
    def ak_97__process_documents(self, ak_97_collection_path: str, ak_97_documents: List[Dict],
                                ak_97_persona: str, ak_97_task: str) -> List[Dict]:

        ak_97_pdf_dir = os.path.join(ak_97_collection_path, "PDFs")
        if not os.path.exists(ak_97_pdf_dir):
            raise FileNotFoundError(f"PDFs directory not found: {ak_97_pdf_dir}")
        
        ak_97_available_pdfs = [f for f in os.listdir(ak_97_pdf_dir) if f.lower().endswith('.pdf')]
        ak_97_all_sections = []
        
        for ak_97_doc in ak_97_documents:
            ak_97_filename = ak_97_doc["filename"]
            ak_97_matched_pdf = self.ak_97__find_matching_pdf(ak_97_filename, ak_97_available_pdfs)
            
            if ak_97_matched_pdf:
                ak_97_pdf_path = os.path.join(ak_97_pdf_dir, ak_97_matched_pdf)
                print(f"✓ Processing: {ak_97_matched_pdf}")
                
                try:
                    # Use enhanced structure analyzer
                    ak_97_title, ak_97_outline, ak_97_full_text = self.ak_97_structure_analyzer.ak_97_process_document_structure(ak_97_pdf_path)
                    
                    # Extract sections with context
                    ak_97_sections = self.ak_97__extract_sections_with_context(
                        ak_97_pdf_path, ak_97_title, ak_97_outline, ak_97_full_text, ak_97_filename, ak_97_persona, ak_97_task
                    )
                    
                    ak_97_all_sections.extend(ak_97_sections)
                    
                except Exception as e:
                    print(f"Error processing {ak_97_matched_pdf}: {e}")
                    continue
            else:
                print(f"✗ Could not find match for '{ak_97_filename}'")
        
        return ak_97_all_sections
    
    def ak_97__find_matching_pdf(self, ak_97_target_filename: str, ak_97_available_pdfs: List[str]) -> Optional[str]:

        # Strategy 1: Exact match
        if ak_97_target_filename in ak_97_available_pdfs:
            return ak_97_target_filename
        
        # Strategy 2: Case-insensitive match
        ak_97_target_lower = ak_97_target_filename.lower()
        for ak_97_pdf in ak_97_available_pdfs:
            if ak_97_pdf.lower() == ak_97_target_lower:
                return ak_97_pdf
        
        # Strategy 3: Fuzzy match (normalize names)
        def ak_97_normalize_name(ak_97_name):
            return re.sub(r'[^a-zA-Z0-9]', '', ak_97_name.lower())
        
        ak_97_target_normalized = ak_97_normalize_name(ak_97_target_filename)
        for ak_97_pdf in ak_97_available_pdfs:
            if ak_97_normalize_name(ak_97_pdf) == ak_97_target_normalized:
                return ak_97_pdf
        
        return None
    
    def ak_97__extract_sections_with_context(self, ak_97_pdf_path: str, ak_97_title: str, ak_97_outline: List[Dict],
                                            ak_97_full_text: str, ak_97_filename: str, ak_97_persona: str, ak_97_task: str) -> List[Dict]:

        ak_97_sections = []
        
        if not ak_97_outline:
            # If no outline detected, create sections from pages
            try:
                ak_97_doc = fitz.open(ak_97_pdf_path)
                for ak_97_page_num in range(len(ak_97_doc)):
                    ak_97_page = ak_97_doc[ak_97_page_num]
                    ak_97_page_text = ak_97_page.get_text()
                    
                    if len(ak_97_page_text.strip()) > 100:
                        ak_97_sections.append({
                            "document": ak_97_filename,
                            "title": f"Content from {ak_97_filename} - Page {ak_97_page_num + 1}",
                            "text": ak_97_page_text.strip(),
                            "page": ak_97_page_num,
                            "level": "content"
                        })
                ak_97_doc.close()
            except Exception as e:
                print(f"Error extracting page content: {e}")
            
            return ak_97_sections
        
        # Extract content for each heading
        try:
            ak_97_doc = fitz.open(ak_97_pdf_path)
            
            for i, ak_97_heading in enumerate(ak_97_outline):
                ak_97_page_num = ak_97_heading["page"]
                ak_97_heading_text = ak_97_heading["text"]
                
                try:
                    ak_97_page = ak_97_doc[ak_97_page_num]
                    ak_97_page_text = ak_97_page.get_text()
                    
                    # Find heading position and extract following content
                    ak_97_heading_pos = ak_97_page_text.lower().find(ak_97_heading_text.lower())
                    
                    if ak_97_heading_pos != -1:
                        ak_97_content_start = ak_97_heading_pos + len(ak_97_heading_text)
                        
                        # Find next heading or end of page
                        ak_97_next_heading_pos = len(ak_97_page_text)
                        for j in range(i + 1, len(ak_97_outline)):
                            if ak_97_outline[j]["page"] == ak_97_page_num:
                                ak_97_next_pos = ak_97_page_text.lower().find(ak_97_outline[j]["text"].lower())
                                if ak_97_next_pos > ak_97_heading_pos:
                                    ak_97_next_heading_pos = ak_97_next_pos
                                    break
                        
                        ak_97_section_text = ak_97_page_text[ak_97_content_start:ak_97_next_heading_pos].strip()
                        
                        # Include substantial sections only
                        if len(ak_97_section_text) > 50:
                            ak_97_sections.append({
                                "document": ak_97_filename,
                                "title": ak_97_heading_text,
                                "text": ak_97_section_text,
                                "page": ak_97_page_num,
                                "level": ak_97_heading["level"]
                            })
                
                except Exception as e:
                    print(f"Error extracting section {ak_97_heading_text}: {e}")
                    continue
            
            ak_97_doc.close()
            
        except Exception as e:
            print(f"Error processing document structure: {e}")
        
        return ak_97_sections
    
    def ak_97__rank_sections_by_relevance(self, ak_97_sections: List[Dict], ak_97_persona: str, ak_97_task: str) -> List[Dict]:

        ak_97_scored_sections = []
        
        for ak_97_section in ak_97_sections:
            # Use SLM for semantic analysis
            ak_97_relevance_score = self.ak_97_slm.ak_97_analyze_content_relevance(
                ak_97_section.get("text", ""), ak_97_persona, ak_97_task
            )
            
            # Add title relevance bonus
            ak_97_title_score = self.ak_97_slm.ak_97_analyze_content_relevance(
                ak_97_section.get("title", ""), ak_97_persona, ak_97_task
            )
            
            # Heading level bonus (higher level = more important)
            ak_97_level_bonus = 0.0
            if ak_97_section.get("level") == "H1":
                ak_97_level_bonus = 0.3
            elif ak_97_section.get("level") == "H2":
                ak_97_level_bonus = 0.2
            elif ak_97_section.get("level") == "H3":
                ak_97_level_bonus = 0.1
            
            # Content length normalization
            ak_97_text_length = len(ak_97_section.get("text", ""))
            ak_97_length_score = min(ak_97_text_length / 1000, 1.0) * 0.2
            
            # Combine scores
            ak_97_total_score = (
                ak_97_relevance_score * 0.5 +
                ak_97_title_score * 0.3 +
                ak_97_level_bonus +
                ak_97_length_score
            )
            
            ak_97_scored_sections.append({
                **ak_97_section,
                "relevance_score": ak_97_total_score
            })
        
        # Sort by relevance score (descending)
        return sorted(ak_97_scored_sections, key=lambda x: x["relevance_score"], reverse=True)
    
    def ak_97__generate_output(self, ak_97_documents: List[Dict], ak_97_persona: str, ak_97_task: str,
                              ak_97_ranked_sections: List[Dict]) -> Dict[str, Any]:

        ak_97_output_data = {
            "metadata": {
                "input_documents": [ak_97_doc["filename"] for ak_97_doc in ak_97_documents],
                "persona": ak_97_persona,
                "job_to_be_done": ak_97_task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Add top ranked sections (limit to top 15)
        for i, ak_97_section in enumerate(ak_97_ranked_sections[:15]):
            ak_97_output_data["extracted_sections"].append({
                "document": ak_97_section["document"],
                "section_title": ak_97_section["title"],
                "importance_rank": i + 1,
                "page_number": ak_97_section["page"] + 1  # Convert to 1-based
            })
        
        # Add subsection analysis for top 8 sections
        for ak_97_section in ak_97_ranked_sections[:8]:
            ak_97_subsections = self.ak_97__analyze_subsections(ak_97_section)
            for ak_97_subsection in ak_97_subsections:
                ak_97_output_data["subsection_analysis"].append({
                    "document": ak_97_section["document"],
                    "refined_text": ak_97_subsection["text"],
                    "page_number": ak_97_section["page"] + 1  # Convert to 1-based
                })
        
        return ak_97_output_data
    
    def ak_97__analyze_subsections(self, ak_97_section: Dict) -> List[Dict]:

        ak_97_text = ak_97_section.get("text", "")
        if not ak_97_text:
            return []
        
        # Split into paragraphs
        ak_97_paragraphs = [ak_97_p.strip() for ak_97_p in ak_97_text.split('\n\n') if ak_97_p.strip()]
        
        # Filter for substantial paragraphs
        ak_97_substantial_paragraphs = [ak_97_p for ak_97_p in ak_97_paragraphs if len(ak_97_p) > 100]
        
        if not ak_97_substantial_paragraphs:
            ak_97_substantial_paragraphs = [ak_97_p for ak_97_p in ak_97_paragraphs if len(ak_97_p) > 50]
        
        ak_97_subsections = []
        for ak_97_para in ak_97_substantial_paragraphs[:3]:  # Top 3 paragraphs
            # Refine text (truncate if too long)
            ak_97_refined_text = ak_97_para[:400] + "..." if len(ak_97_para) > 400 else ak_97_para
            ak_97_subsections.append({
                "text": ak_97_refined_text
            })
        
        return ak_97_subsections

def ak_97_main():
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <collection_path>")
        sys.exit(1)
    
    ak_97_collection_path = os.path.abspath(sys.argv[1])
    
    if not os.path.exists(ak_97_collection_path):
        print(f"Error: Collection path does not exist: {ak_97_collection_path}")
        sys.exit(1)
    
    try:
        ak_97_analyzer = ak_97_PersonaDrivenAnalyzer()
        ak_97_result = ak_97_analyzer.ak_97_process_collection(ak_97_collection_path)
        

        ak_97_output_path = os.path.join(ak_97_collection_path, "challenge1b_output.json")
        with open(ak_97_output_path, 'w', encoding='utf-8') as f:
            json.dump(ak_97_result, f, indent=4, ensure_ascii=False)
        
        print(f"\n✓ Successfully processed collection at {ak_97_collection_path}")
        print(f"✓ Processed {len(ak_97_result['metadata']['input_documents'])} documents")
        print(f"✓ Extracted {len(ak_97_result['extracted_sections'])} relevant sections")
        print(f"✓ Generated {len(ak_97_result['subsection_analysis'])} subsection analyses")
        print(f"✓ Output saved to: {ak_97_output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    ak_97_main()
