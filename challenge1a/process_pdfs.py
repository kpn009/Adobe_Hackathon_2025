import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional
import fitz
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
        
        self.ak_97_european_languages = {
            "en", "de", "fr", "es", "it", "pt", "nl", "sv", "no", "da", "fi",
            "is", "fo", "pl", "cs", "sk", "sl", "hr", "bs", "sq", "ro", "hu",
            "et", "lv", "lt", "el", "mt", "ga", "gd", "cy", "br", "eu", "ca",
            "gl", "oc", "co", "sc", "rm", "fur", "lld", "vec", "scn", "nap",
            "lij", "pms", "lmo", "eml", "rgn", "an", "ast", "ext", "mwl", "mirandese"
        }
        
        self.ak_97_asian_languages = {
            "zh", "ja", "ko", "th", "vi", "id", "ms", "tl", "ceb", "hil", "war",
            "bcl", "pag", "pam", "bik", "ilo", "my", "km", "lo", "bo", "dz", "mn",
            "kk", "ky", "uz", "tg", "tk", "az", "tr", "hy", "ka", "am", "ti", "om"
        }
        
        self.ak_97_southeast_asian_languages = {
            "vi", "id", "ms", "tl", "th", "my", "km", "lo", "ceb", "hil", "war",
            "bcl", "pag", "pam", "bik", "ilo", "jv", "su", "mad", "bug", "min"
        }
        
        self.ak_97__prepare_filter_patterns()
    
    def ak_97__prepare_filter_patterns(self):
        self.ak_97_content_filters = [
            re.compile(r'(?i)copyright\s*[©®™]?'),
            re.compile(r'(?i)©\s*\d{4}'),
            re.compile(r'(?i)version\s*\d+'),
            re.compile(r'(?i)v\.\s*\d+'),
            re.compile(r'(?i)edition\s*\d*'),
            re.compile(r'(?i)published\s+(by|in)'),
            re.compile(r'(?i)isbn[\s:-]*\d'),
            re.compile(r'(?i)all\s+rights\s+reserved'),
            re.compile(r'(?i)trademark\s*[™®]?'),
            re.compile(r'(?i)www\.\w+\.\w+'),
            re.compile(r'(?i)http[s]?://\S+'),
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        ]
        
        self.ak_97_contact_web_filters = [
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            re.compile(r'\+\d{1,3}[\s.-]?\d{1,14}'),
            re.compile(r'\b\d{10,15}\b'),
            re.compile(r'(?i)https?://[^\s]+'),
            re.compile(r'(?i)www\.[^\s]+'),
            re.compile(r'(?i)ftp://[^\s]+'),
        ]
        
        self.ak_97_field_label_filter = re.compile(r'^\d+\.\s*$|^\w+\.\s*$|^[A-Z][a-z]*\s+of\s+\w+\s*[.:]?\s*$')
    
    def ak_97_clean_text_for_matching(self, ak_97_input_text: str) -> str:
        ak_97_cleaned_chars = []
        for ak_97_character in ak_97_input_text:
            ak_97_char_category = unicodedata.category(ak_97_character)
            if ak_97_char_category[0] in ("L", "N"):
                ak_97_char_name = unicodedata.name(ak_97_character, "")
                if any(ak_97_script in ak_97_char_name.upper() for ak_97_script in ["LATIN", "CYRILLIC", "GREEK"]):
                    ak_97_cleaned_chars.append(ak_97_character.lower())
                else:
                    ak_97_cleaned_chars.append(ak_97_character)
        return "".join(ak_97_cleaned_chars)
    
    def ak_97_identify_language(self, ak_97_input_text: str) -> str:
        if not ak_97_input_text or not ak_97_input_text.strip():
            return "en"
        ak_97_detected_lang, _ = langid.classify(ak_97_input_text)
        return ak_97_detected_lang
    
    def ak_97_measure_content_units(self, ak_97_input_text: str, ak_97_lang_code: str) -> int:
        if ak_97_lang_code in self.ak_97_character_based_languages:
            return len(ak_97_input_text.strip())
        elif ak_97_lang_code in self.ak_97_arabic_script_languages:
            return len([ak_97_char for ak_97_char in ak_97_input_text.strip() if unicodedata.category(ak_97_char)[0] == 'L'])
        else:
            return len(ak_97_input_text.strip().split())
    
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
    
    def ak_97_analyze_text_style(self, ak_97_text_line: Dict) -> Dict:
        if not ak_97_text_line.get("spans"):
            return {"size": 0, "bold": False, "italic": False, "font": "", "color": 0}
        
        ak_97_text_span = ak_97_text_line["spans"][0]
        ak_97_style_flags = ak_97_text_span.get("flags", 0)
        ak_97_typeface_name = ak_97_text_span.get("font", "").lower()
        
        ak_97_weight_indicators = ["bold", "bd", "blk", "black", "heavy", "fett", "gras", "negrita", "grassetto", "negrito"]
        ak_97_has_bold_weight = (ak_97_style_flags & 2**4) != 0 or any(ak_97_indicator in ak_97_typeface_name for ak_97_indicator in ak_97_weight_indicators)
        
        ak_97_slope_indicators = ["italic", "oblique", "slant", "kursiv", "italique", "cursiva", "corsivo", "italico"]
        ak_97_has_italic_slope = (ak_97_style_flags & 2**6) != 0 or any(ak_97_indicator in ak_97_typeface_name for ak_97_indicator in ak_97_slope_indicators)
        
        return {
            "size": round(ak_97_text_span.get("size", 0)),
            "bold": ak_97_has_bold_weight,
            "italic": ak_97_has_italic_slope,
            "font": ak_97_typeface_name,
            "color": 0
        }
    
    def ak_97_compare_text_styles(self, ak_97_first_style: Dict, ak_97_second_style: Dict, ak_97_tolerance: int = 1) -> bool:
        return (
            abs(ak_97_first_style["size"] - ak_97_second_style["size"]) <= ak_97_tolerance and
            ak_97_first_style["bold"] == ak_97_second_style["bold"] and
            ak_97_first_style["italic"] == ak_97_second_style["italic"] and
            ak_97_first_style["font"] == ak_97_second_style["font"]
        )
    
    def ak_97_matches_excluded_patterns(self, ak_97_input_text: str) -> bool:
        return any(ak_97_filter_pattern.search(ak_97_input_text) for ak_97_filter_pattern in self.ak_97_content_filters)
    
    def ak_97_has_contact_or_web_info(self, ak_97_input_text: str) -> bool:
        return any(ak_97_filter_pattern.search(ak_97_input_text) for ak_97_filter_pattern in self.ak_97_contact_web_filters)
    
    def ak_97_check_section_text_validity(self, ak_97_input_text: str) -> bool:
        ak_97_cleaned_text = ak_97_input_text.strip()
        
        if len(ak_97_cleaned_text) < 2:
            return False
        
        if ak_97_cleaned_text.endswith(('.', ',', ';', ':', '!', '?', '"', "'", ')', ']', '}', '。', '！', '？', '；', '：', '»', '«', '‹', '›')) and not ak_97_cleaned_text.endswith('...'):
            return False
        
        if self.ak_97_field_label_filter.match(ak_97_cleaned_text):
            return False
        
        if self.ak_97_matches_excluded_patterns(ak_97_cleaned_text):
            return False
        
        if self.ak_97_has_contact_or_web_info(ak_97_cleaned_text):
            return False
        
        return True
    
    def ak_97_locate_subsequent_content(self, ak_97_all_text_lines: List[Dict], ak_97_current_index: int,
                                       ak_97_main_text_size: int, ak_97_min_content_threshold: int,
                                       ak_97_lang_code: str) -> bool:
        ak_97_reference_line = ak_97_all_text_lines[ak_97_current_index]
        ak_97_reference_page = ak_97_reference_line["page_num"]
        ak_97_lookup_boundary = min(ak_97_current_index + 8, len(ak_97_all_text_lines))
        
        ak_97_valid_content_count = 0
        ak_97_sequential_valid = 0
        
        for ak_97_next_position in range(ak_97_current_index + 1, ak_97_lookup_boundary):
            ak_97_following_line = ak_97_all_text_lines[ak_97_next_position]
            
            if ak_97_following_line["page_num"] != ak_97_reference_page:
                break
            
            ak_97_following_content = ak_97_following_line.get("text", "").strip()
            if len(ak_97_following_content) < 3:
                continue
            
            ak_97_content_unit_count = self.ak_97_measure_content_units(ak_97_following_content, ak_97_lang_code)
            ak_97_text_size = round(ak_97_following_line["size"])
            
            if ak_97_text_size > ak_97_reference_line["size"] + 2:
                break
            
            ak_97_min_length_threshold = ak_97_min_content_threshold if ak_97_lang_code not in self.ak_97_character_based_languages else ak_97_min_content_threshold * 2
            
            if (abs(ak_97_text_size - ak_97_main_text_size) <= 1 and
                ak_97_content_unit_count >= ak_97_min_content_threshold and
                len(ak_97_following_content) >= ak_97_min_length_threshold):
                ak_97_valid_content_count += 1
                ak_97_sequential_valid += 1
                
                if ak_97_sequential_valid >= 1 and ak_97_content_unit_count >= ak_97_min_content_threshold * 1.5:
                    return True
            else:
                ak_97_sequential_valid = 0
            
            if ak_97_next_position - ak_97_current_index > 4 and ak_97_valid_content_count == 0:
                break
        
        if ak_97_valid_content_count == 0:
            if self.ak_97__verify_organized_content(ak_97_all_text_lines, ak_97_current_index, ak_97_lookup_boundary,
                                                   ak_97_reference_page, ak_97_main_text_size, ak_97_min_content_threshold, ak_97_lang_code):
                return True
        
        return self.ak_97__verify_combined_content(ak_97_all_text_lines, ak_97_current_index, ak_97_reference_page,
                                                  ak_97_main_text_size, ak_97_min_content_threshold, ak_97_lang_code)
    
    def ak_97__verify_organized_content(self, ak_97_all_text_lines: List[Dict], ak_97_current_index: int,
                                       ak_97_lookup_boundary: int, ak_97_reference_page: int, ak_97_main_text_size: int,
                                       ak_97_min_content_threshold: int, ak_97_lang_code: str) -> bool:
        ak_97_reference_line = ak_97_all_text_lines[ak_97_current_index]
        ak_97_position_markers = []
        ak_97_organized_lines = []
        
        for ak_97_next_position in range(ak_97_current_index + 1, ak_97_lookup_boundary):
            ak_97_following_line = ak_97_all_text_lines[ak_97_next_position]
            
            if ak_97_following_line["page_num"] != ak_97_reference_page:
                break
            
            ak_97_following_content = ak_97_following_line.get("text", "").strip()
            ak_97_min_text_length = 5 if ak_97_lang_code not in self.ak_97_character_based_languages else 3
            
            if len(ak_97_following_content) < ak_97_min_text_length or ak_97_following_line["size"] > ak_97_reference_line["size"] + 1:
                continue
            
            if abs(round(ak_97_following_line["size"]) - ak_97_main_text_size) <= 2:
                ak_97_content_unit_count = self.ak_97_measure_content_units(ak_97_following_content, ak_97_lang_code)
                
                if ak_97_content_unit_count >= max(3, ak_97_min_content_threshold // 2):
                    if ak_97_following_line.get("spans") and len(ak_97_following_line["spans"]) > 0:
                        ak_97_horizontal_pos = round(ak_97_following_line["spans"][0]["bbox"][0])
                        ak_97_adjusted_pos = round(ak_97_horizontal_pos / 15) * 15
                        ak_97_position_markers.append(ak_97_adjusted_pos)
                        ak_97_organized_lines.append(ak_97_following_line)
            
            if len(ak_97_organized_lines) >= 5:
                break
        
        if len(ak_97_organized_lines) >= 3:
            ak_97_position_frequency = Counter(ak_97_position_markers)
            ak_97_highest_frequency = ak_97_position_frequency.most_common(1)[0][1]
            ak_97_distinct_positions = len(ak_97_position_frequency)
            
            if ak_97_highest_frequency >= 3 or (ak_97_distinct_positions >= 2 and ak_97_highest_frequency >= 2):
                ak_97_combined_text = " ".join(ak_97_text_line.get("text", "") for ak_97_text_line in ak_97_organized_lines)
                return len(ak_97_combined_text) >= ak_97_min_content_threshold * 3
        
        return False
    
    def ak_97__verify_combined_content(self, ak_97_all_text_lines: List[Dict], ak_97_current_index: int,
                                      ak_97_reference_page: int, ak_97_main_text_size: int,
                                      ak_97_min_content_threshold: int, ak_97_lang_code: str) -> bool:
        ak_97_reference_line = ak_97_all_text_lines[ak_97_current_index]
        ak_97_aggregate_content_length = 0
        ak_97_meaningful_lines = 0
        
        for ak_97_next_position in range(ak_97_current_index + 1, min(ak_97_current_index + 6, len(ak_97_all_text_lines))):
            ak_97_following_line = ak_97_all_text_lines[ak_97_next_position]
            
            if ak_97_following_line["page_num"] != ak_97_reference_page:
                break
            
            ak_97_following_content = ak_97_following_line.get("text", "").strip()
            ak_97_min_text_length = 5 if ak_97_lang_code not in self.ak_97_character_based_languages else 3
            
            if len(ak_97_following_content) < ak_97_min_text_length or ak_97_following_line["size"] > ak_97_reference_line["size"] + 1:
                continue
            
            ak_97_punctuation_endings = [':'] if ak_97_lang_code in self.ak_97_european_languages else [':', '：']
            
            if (abs(round(ak_97_following_line["size"]) - ak_97_main_text_size) <= 2 and
                not any(ak_97_following_content.endswith(p) for p in ak_97_punctuation_endings) and
                len(ak_97_following_content) >= ak_97_min_text_length * 2):
                
                ak_97_content_length = self.ak_97_measure_content_units(ak_97_following_content, ak_97_lang_code)
                ak_97_aggregate_content_length += ak_97_content_length
                ak_97_meaningful_lines += 1
                
                if (ak_97_meaningful_lines >= 2 and
                    ak_97_aggregate_content_length >= ak_97_min_content_threshold * 2 and
                    ak_97_content_length >= ak_97_min_content_threshold):
                    return True
        
        return False
    
    def ak_97_derive_document_title(self, ak_97_file_document: fitz.Document) -> str:
        try:
            ak_97_opening_page = ak_97_file_document[0]
            ak_97_content_blocks = ak_97_opening_page.get_text("dict", flags=4)["blocks"]
            ak_97_extracted_lines = self.ak_97__collect_lines_from_blocks(ak_97_content_blocks)
            
            if not ak_97_extracted_lines:
                return ""
            
            ak_97_primary_title = self.ak_97__determine_title_candidate(ak_97_extracted_lines, 0.90, 1.5, False)
            
            if len(ak_97_primary_title) < 15:
                ak_97_secondary_title = self.ak_97__determine_title_candidate(ak_97_extracted_lines, 0.80, 1.8, True)
                if len(ak_97_secondary_title) > len(ak_97_primary_title):
                    return ak_97_secondary_title
            
            return ak_97_primary_title
            
        except Exception:
            return ""
    
    def ak_97__collect_lines_from_blocks(self, ak_97_content_blocks: List[Dict]) -> List[Dict]:
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
        
        return ak_97_extracted_lines
    
    def ak_97__determine_title_candidate(self, ak_97_text_lines: List[Dict], ak_97_size_ratio: float,
                                        ak_97_spacing_multiplier: float, ak_97_select_optimal_block: bool) -> str:
        if not ak_97_text_lines:
            return ""
        
        ak_97_largest_size = max(ak_97_text_line.get("size", 0) for ak_97_text_line in ak_97_text_lines)
        if ak_97_largest_size == 0:
            return ""
        
        ak_97_minimum_size = ak_97_largest_size * ak_97_size_ratio
        ak_97_potential_titles = [ak_97_text_line for ak_97_text_line in ak_97_text_lines if ak_97_text_line.get("size", 0) >= ak_97_minimum_size]
        
        if not ak_97_potential_titles:
            return ""
        
        ak_97_potential_titles.sort(key=lambda x: x["y0"])
        ak_97_grouped_blocks = self.ak_97__organize_lines_into_clusters(ak_97_potential_titles, ak_97_spacing_multiplier)
        
        if not ak_97_grouped_blocks:
            return ""
        
        ak_97_selected_block = (
            max(ak_97_grouped_blocks, key=lambda ak_97_cluster: (len(ak_97_cluster), -ak_97_cluster[0]["y0"]))
            if ak_97_select_optimal_block else ak_97_grouped_blocks[0]
        )
        
        ak_97_constructed_title = " ".join(ak_97_text_line["text"] for ak_97_text_line in ak_97_selected_block).strip()
        
        if (len(ak_97_constructed_title.split()) > self.ak_97_settings["MAX_DOCUMENT_TITLE_WORDS"] or
            ak_97_constructed_title.endswith(("!", "！"))):
            return ""
        
        return ak_97_constructed_title
    
    def ak_97__organize_lines_into_clusters(self, ak_97_potential_titles: List[Dict], ak_97_spacing_multiplier: float) -> List[List[Dict]]:
        if not ak_97_potential_titles:
            return []
        
        ak_97_line_clusters = []
        ak_97_active_cluster = [ak_97_potential_titles[0]]
        
        for i in range(1, len(ak_97_potential_titles)):
            ak_97_preceding_line, ak_97_current_line = ak_97_active_cluster[-1], ak_97_potential_titles[i]
            ak_97_spacing_gap = ak_97_current_line["y0"] - ak_97_preceding_line["y1"]
            ak_97_spacing_limit = ak_97_preceding_line["size"] * ak_97_spacing_multiplier
            
            if ak_97_spacing_gap < ak_97_spacing_limit:
                ak_97_active_cluster.append(ak_97_current_line)
            else:
                ak_97_line_clusters.append(ak_97_active_cluster)
                ak_97_active_cluster = [ak_97_current_line]
        
        ak_97_line_clusters.append(ak_97_active_cluster)
        return ak_97_line_clusters
    
    def ak_97_build_structure_from_analysis(self, ak_97_file_document: fitz.Document,
                                           ak_97_document_title: str = "", ak_97_lang_code: str = "en") -> List[Dict]:
        ak_97_extracted_text_lines = self.ak_97__gather_all_text_lines(ak_97_file_document)
        if not ak_97_extracted_text_lines:
            return []
        
        ak_97_cleaned_title = self.ak_97_clean_text_for_matching(ak_97_document_title) if ak_97_document_title else ""
        ak_97_standard_text_size = self.ak_97__establish_standard_text_size(ak_97_extracted_text_lines, ak_97_lang_code)
        ak_97_min_content_threshold = self.ak_97_retrieve_min_content_threshold(ak_97_lang_code)
        ak_97_document_height = ak_97_file_document[0].rect.height
        
        ak_97_initial_section_candidates = self.ak_97__discover_section_candidates(
            ak_97_extracted_text_lines, ak_97_standard_text_size, ak_97_min_content_threshold, ak_97_document_height,
            ak_97_cleaned_title, ak_97_lang_code
        )
        
        if not ak_97_initial_section_candidates:
            return []
        
        ak_97_supplementary_sections = self.ak_97__discover_style_matched_sections(
            ak_97_extracted_text_lines, ak_97_initial_section_candidates, ak_97_document_height,
            ak_97_cleaned_title, ak_97_lang_code
        )
        
        ak_97_combined_sections = ak_97_initial_section_candidates + ak_97_supplementary_sections
        return self.ak_97__organize_and_combine_sections(ak_97_combined_sections)
    
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
        ak_97_main_text_candidates = [ak_97_text_line for ak_97_text_line in ak_97_collected_lines if len(ak_97_text_line["text"]) > ak_97_threshold_length]
        
        if not ak_97_main_text_candidates:
            ak_97_main_text_candidates = ak_97_collected_lines
        
        ak_97_size_distribution = Counter(round(ak_97_text_line["size"]) for ak_97_text_line in ak_97_main_text_candidates)
        return ak_97_size_distribution.most_common(1)[0][0] if ak_97_size_distribution else 12
    
    def ak_97__discover_section_candidates(self, ak_97_collected_lines: List[Dict], ak_97_standard_text_size: int,
                                          ak_97_min_content_threshold: int, ak_97_document_height: float,
                                          ak_97_cleaned_title: str, ak_97_lang_code: str) -> List[Dict]:
        ak_97_section_candidates = []
        ak_97_boundary_margin = self.ak_97_settings["MARGIN_BOUNDARY"]
        
        for i, ak_97_text_line in enumerate(ak_97_collected_lines):
            if not self.ak_97__evaluate_as_potential_section(ak_97_text_line, ak_97_standard_text_size, ak_97_document_height,
                                                            ak_97_cleaned_title, ak_97_lang_code, ak_97_boundary_margin):
                continue
            
            if self.ak_97_locate_subsequent_content(ak_97_collected_lines, i, ak_97_standard_text_size,
                                                   ak_97_min_content_threshold, ak_97_lang_code):
                ak_97_section_candidates.append(ak_97_text_line)
        
        return ak_97_section_candidates
    
    def ak_97__evaluate_as_potential_section(self, ak_97_text_line: Dict, ak_97_standard_text_size: int, ak_97_document_height: float,
                                            ak_97_cleaned_title: str, ak_97_lang_code: str, ak_97_boundary_margin: float) -> bool:
        if not self.ak_97_check_section_text_validity(ak_97_text_line["text"]):
            return False
        
        ak_97_content_unit_count = self.ak_97_measure_content_units(ak_97_text_line["text"], ak_97_lang_code)
        ak_97_text_style = self.ak_97_analyze_text_style(ak_97_text_line)
        
        ak_97_within_margins = (ak_97_text_line["bbox"][1] < ak_97_boundary_margin * ak_97_document_height or
                               ak_97_text_line["bbox"][1] > (1 - ak_97_boundary_margin) * ak_97_document_height)
        
        ak_97_matches_title = (ak_97_cleaned_title and
                              self.ak_97_clean_text_for_matching(ak_97_text_line["text"]) in ak_97_cleaned_title)
        
        if ak_97_within_margins or ak_97_matches_title:
            return False
        
        ak_97_max_units = self.ak_97_settings["MAX_SECTION_WORDS"]
        if ak_97_lang_code in self.ak_97_character_based_languages:
            ak_97_max_units = ak_97_max_units * 2
        
        ak_97_size_qualifies_as_section = (
            ak_97_text_line["size"] >= self.ak_97_settings["TITLE_SIZE_MULTIPLIER"] * ak_97_standard_text_size and
            1 <= ak_97_content_unit_count <= ak_97_max_units
        )
        
        ak_97_weight_qualifies_as_section = (
            ak_97_text_style["bold"] and
            ak_97_standard_text_size - 1 <= ak_97_text_style["size"] <= ak_97_standard_text_size + 2 and
            2 <= ak_97_content_unit_count <= ak_97_max_units and
            len(ak_97_text_line["text"]) > 2
        )
        
        return ak_97_size_qualifies_as_section or ak_97_weight_qualifies_as_section
    
    def ak_97__discover_style_matched_sections(self, ak_97_collected_lines: List[Dict], ak_97_section_candidates: List[Dict],
                                              ak_97_document_height: float, ak_97_cleaned_title: str,
                                              ak_97_lang_code: str) -> List[Dict]:
        ak_97_established_styles = []
        for ak_97_section in ak_97_section_candidates:
            ak_97_text_style = self.ak_97_analyze_text_style(ak_97_section)
            if ak_97_text_style["size"] > 0:
                ak_97_established_styles.append(ak_97_text_style)
        
        if not ak_97_established_styles:
            return []
        
        ak_97_existing_section_texts = set(ak_97_text_line["text"] for ak_97_text_line in ak_97_section_candidates)
        ak_97_additional_sections = []
        ak_97_boundary_margin = self.ak_97_settings["MARGIN_BOUNDARY"]
        
        for ak_97_text_line in ak_97_collected_lines:
            if ak_97_text_line["text"] in ak_97_existing_section_texts:
                continue
            
            ak_97_current_style = self.ak_97_analyze_text_style(ak_97_text_line)
            if ak_97_current_style["size"] == 0:
                continue
            
            ak_97_style_matches_existing = any(
                self.ak_97_compare_text_styles(ak_97_current_style, ak_97_established_style, tolerance=1)
                for ak_97_established_style in ak_97_established_styles
            )
            
            if not ak_97_style_matches_existing:
                continue
            
            ak_97_content_unit_count = self.ak_97_measure_content_units(ak_97_text_line["text"], ak_97_lang_code)
            ak_97_within_margins = (ak_97_text_line["bbox"][1] < ak_97_boundary_margin * ak_97_document_height or
                                   ak_97_text_line["bbox"][1] > (1 - ak_97_boundary_margin) * ak_97_document_height)
            
            ak_97_matches_title = (ak_97_cleaned_title and
                                  self.ak_97_clean_text_for_matching(ak_97_text_line["text"]) in ak_97_cleaned_title)
            
            ak_97_max_units = self.ak_97_settings["MAX_SECTION_WORDS"]
            if ak_97_lang_code in self.ak_97_character_based_languages:
                ak_97_max_units = ak_97_max_units * 2
            
            ak_97_qualifies_as_section = (
                2 <= ak_97_content_unit_count <= ak_97_max_units and
                len(ak_97_text_line["text"]) > 2 and
                not ak_97_within_margins and
                not ak_97_matches_title and
                self.ak_97_check_section_text_validity(ak_97_text_line["text"])
            )
            
            if ak_97_qualifies_as_section:
                ak_97_additional_sections.append(ak_97_text_line)
                ak_97_existing_section_texts.add(ak_97_text_line["text"])
        
        return ak_97_additional_sections
    
    def ak_97__organize_and_combine_sections(self, ak_97_combined_sections: List[Dict]) -> List[Dict]:
        if not ak_97_combined_sections:
            return []
        
        ak_97_distinct_sizes = set(round(ak_97_section["size"]) for ak_97_section in ak_97_combined_sections)
        ak_97_ordered_sizes = sorted(list(ak_97_distinct_sizes), reverse=True)
        
        ak_97_size_level_mapping = {
            ak_97_text_size: f"H{i+1}"
            for i, ak_97_text_size in enumerate(ak_97_ordered_sizes[:self.ak_97_settings["MAX_SECTION_LEVELS"]])
        }
        
        ak_97_structured_sections = []
        for ak_97_section in ak_97_combined_sections:
            ak_97_text_size = round(ak_97_section["size"])
            ak_97_section_level = ak_97_size_level_mapping.get(ak_97_text_size)
            if ak_97_section_level:
                ak_97_structured_sections.append({
                    "level": ak_97_section_level,
                    "text": ak_97_section["text"],
                    "page": ak_97_section["page_num"],
                    "bbox": ak_97_section["bbox"]
                })
        
        if not ak_97_structured_sections:
            return []
        
        ak_97_structured_sections.sort(key=lambda h: (h["page"], h["bbox"][1]))
        return self.ak_97__combine_adjacent_sections(ak_97_structured_sections)
    
    def ak_97__combine_adjacent_sections(self, ak_97_structured_sections: List[Dict]) -> List[Dict]:
        if not ak_97_structured_sections:
            return []
        
        ak_97_combined_sections = [ak_97_structured_sections[0]]
        
        for i in range(1, len(ak_97_structured_sections)):
            ak_97_preceding, ak_97_current = ak_97_combined_sections[-1], ak_97_structured_sections[i]
            
            ak_97_same_page_location = ak_97_current["page"] == ak_97_preceding["page"]
            ak_97_same_hierarchy_level = ak_97_current["level"] == ak_97_preceding["level"]
            ak_97_vertically_adjacent = (
                -2 <= (ak_97_current["bbox"][1] - ak_97_preceding["bbox"][3]) <
                (ak_97_preceding["bbox"][3] - ak_97_preceding["bbox"][1]) * 0.75
            )
            
            if ak_97_same_page_location and ak_97_same_hierarchy_level and ak_97_vertically_adjacent:
                ak_97_preceding["text"] = ak_97_preceding["text"].rstrip() + " " + ak_97_current["text"]
            else:
                ak_97_combined_sections.append(ak_97_current)
        
        return [{"level": h["level"], "text": h["text"], "page": h["page"]} for h in ak_97_combined_sections]
    
    def ak_97_process_document_structure(self, ak_97_file_path: Path) -> Tuple[str, List[Dict]]:
        try:
            ak_97_file_document = fitz.open(ak_97_file_path)
        except Exception:
            return "", []
        
        ak_97_preview_text = "".join(
            ak_97_file_document[i].get_text() for i in range(min(3, len(ak_97_file_document)))
        )
        
        ak_97_detected_language = self.ak_97_identify_language(ak_97_preview_text)
        ak_97_document_title = self.ak_97_derive_document_title(ak_97_file_document)
        
        ak_97_built_in_contents = ak_97_file_document.get_toc(simple=True)
        if ak_97_built_in_contents:
            ak_97_structure_outline = self.ak_97__handle_built_in_contents(ak_97_built_in_contents)
        else:
            ak_97_structure_outline = self.ak_97_build_structure_from_analysis(
                ak_97_file_document, document_title=ak_97_document_title, lang_code=ak_97_detected_language
            )
        
        ak_97_validated_outline = self.ak_97__confirm_section_validity(ak_97_file_document, ak_97_structure_outline, ak_97_document_title)
        ak_97_file_document.close()
        
        return ak_97_document_title, ak_97_validated_outline
    
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
    
    def ak_97__confirm_section_validity(self, ak_97_file_document: fitz.Document, ak_97_structure_outline: List[Dict],
                                       ak_97_document_title: str) -> List[Dict]:
        ak_97_normalized_page_texts = {
            i: self.ak_97_clean_text_for_matching(ak_97_document_page.get_text())
            for i, ak_97_document_page in enumerate(ak_97_file_document)
        }
        
        ak_97_cleaned_title = self.ak_97_clean_text_for_matching(ak_97_document_title) if ak_97_document_title else ""
        ak_97_confirmed_outline = []
        
        for ak_97_section in ak_97_structure_outline:
            ak_97_target_page = ak_97_section["page"]
            ak_97_normalized_section_text = self.ak_97_clean_text_for_matching(ak_97_section["text"])
            
            ak_97_exists_on_target_page = (
                ak_97_target_page in ak_97_normalized_page_texts and
                ak_97_normalized_section_text in ak_97_normalized_page_texts[ak_97_target_page]
            )
            
            ak_97_overlaps_with_title = (
                ak_97_document_title and ak_97_normalized_section_text in ak_97_cleaned_title
            )
            
            if ak_97_exists_on_target_page and not ak_97_overlaps_with_title:
                ak_97_confirmed_outline.append(ak_97_section)
        
        return ak_97_confirmed_outline

def ak_97_execute_document_processing():
    ak_97_structure_analyzer = ak_97_DocumentStructureAnalyzer()
    import os
    if os.path.exists("/app/input"):
        ak_97_source_directory = Path("/app/input")
        ak_97_output_directory = Path("/app/output")
    else:
        ak_97_source_directory = Path("./input")
        ak_97_output_directory = Path("./output")
    
    ak_97_source_directory.mkdir(exist_ok=True)
    ak_97_output_directory.mkdir(exist_ok=True)
    
    ak_97_document_files = list(ak_97_source_directory.glob("*.pdf"))
    
    if not ak_97_document_files:
        return
    
    for ak_97_document_file in ak_97_document_files:
        ak_97_document_title, ak_97_structure_outline = ak_97_structure_analyzer.ak_97_process_document_structure(ak_97_document_file)
        
        ak_97_result_data = {"title": ak_97_document_title, "outline": ak_97_structure_outline}
        ak_97_result_path = ak_97_output_directory / f"{ak_97_document_file.stem}.json"
        
        with open(ak_97_result_path, "w", encoding="utf-8") as ak_97_output_file:
            json.dump(ak_97_result_data, ak_97_output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    ak_97_execute_document_processing()
