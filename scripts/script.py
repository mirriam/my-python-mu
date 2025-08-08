import pandas as pd
import requests
from bs4 import BeautifulSoup
import base64
import time
import re
import hashlib
import nltk
from requests.exceptions import RequestException
import json
import os
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import torch

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='debug.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Download NLTK punkt_tab and averaged_perceptron_tagger if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Initialize language tool
tool = language_tool_python.LanguageTool('en-US')

# Initialize model and tokenizer
device = torch.device("cpu")  # Always CPU
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.to(device)

# Initialize sentence transformer on CPU
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Constants
MAX_TOTAL_TOKENS = 3000
MAX_RETURN_SEQUENCES = 4
WP_URL = "https://mauritius.mimusjobs.com/wp-json/wp/v2/job-listings"
WP_COMPANY_URL = "https://mauritius.mimusjobs.com/wp-json/wp/v2/company"
WP_MEDIA_URL = "https://mauritius.mimusjobs.com/wp-json/wp/v2/media"
WP_USERNAME = "admin"
WP_APP_PASSWORD = "Xljs I1VY 7XL0 F45N 3Wsv 5qcv"
PROCESSED_IDS_FILE = "mauritius_processed_job_ids.csv"
LAST_PAGE_FILE = "last_processed_page.txt"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'
}
JOB_TYPE_MAPPING = {
    "Full-time": "full-time",
    "Part-time": "part-time",
    "Contract": "contract",
    "Temporary": "temporary",
    "Freelance": "freelance",
    "Internship": "internship",
    "Volunteer": "volunteer"
}

# Utility functions
def sanitize_text(text, is_url=False, is_email=False):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""
    if is_url or is_email:
        text = re.sub(r'[\r\t\f\v]', '', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'—+', '', text)
    text = re.sub(r'←+', '', text)
    text = re.sub(r'[^\x20-\x7E\n\u00C0-\u017F]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text).strip()
    return text

def clean_description(text):
    try:
        matches = tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text
    except Exception as e:
        logger.error(f"Error in grammar correction: {str(e)}")
        return text

def is_good_paraphrase(original: str, candidate: str) -> float:
    try:
        embeddings = similarity_model.encode([original, candidate], convert_to_tensor=True)
        sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return sim_score
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0

def is_grammatically_correct(text):
    matches = tool.check(text)
    return len(matches) < 3

def extract_nouns(text):
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        return nouns
    except Exception as e:
        logger.error(f"Error extracting nouns from {text}: {str(e)}")
        return []

def contains_nouns(paraphrase, required_nouns):
    if not required_nouns:
        return True
    paraphrase_lower = paraphrase.lower()
    return all(noun.lower() in paraphrase_lower for noun in required_nouns)

def extract_capitalized_words(text):
    words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
    return {word: word for word in words if len(word) > 1}

def restore_capitalization(paraphrased, capitalized_words):
    paraphrased_lower = paraphrased.lower()
    result = paraphrased
    for lower_word, orig_word in capitalized_words.items():
        pattern = r'\b' + re.escape(lower_word) + r'\b'
        result = re.sub(pattern, orig_word, result, flags=re.IGNORECASE)
    return result

# Paraphrasing functions
def paraphrase_strict_title(title, max_attempts=3, max_sub_attempts=2):
    def has_repetitions(text):
        tokens = text.lower().split()
        seen = set()
        for i in range(len(tokens) - 2):
            ngram = tuple(tokens[i:i + 3])
            if ngram in seen:
                return True
            seen.add(ngram)
        return False

    def contains_banned_phrase(text, banned_list):
        text_lower = text.lower()
        critical_phrases = [
            "Rewrite the following", "Paraphrased title", "Professionally rewrite",
            "Keep it short", "Use different phrasing", "Short (5–12 words)",
            "Paraphrase", "Paraphrased", "Paraphrasing", "Paraphrased version",
            "Summary", "Summarised", "Summarized", "Summarizing", "Summarising", "None.", "None", "none",
            ".", ":"
        ]
        for phrase in critical_phrases:
            if phrase.lower() in text_lower:
                start_idx = text_lower.find(phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(text), start_idx + len(phrase) + 20)
                context_snippet = text[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(text):
                    context_snippet = context_snippet + "..."
                return True, phrase, context_snippet
        return False, None, None

    def score_paraphrase(original, paraphrased, target_wc):
        sim = is_good_paraphrase(original, paraphrased)
        wc = len(paraphrased.split())
        length_penalty = abs(wc - target_wc) / max(target_wc, 1)
        return (sim + (1 - length_penalty)) / 2, sim, wc

    clean_title = sanitize_text(title)
    if not clean_title:
        logger.error("Input title is empty after sanitization.")
        return title

    nouns = extract_nouns(clean_title)
    capitalized_words = extract_capitalized_words(clean_title)
    nouns_str = ", ".join(nouns) if nouns else "none"
    logger.debug(f"Extracted nouns from title '{clean_title}': {nouns}")
    logger.debug(f"Extracted capitalized words from title '{clean_title}': {list(capitalized_words.values())}")

    prompt = (
        f"Rewrite the following job title professionally, using different phrasing while preserving the meaning. "
        f"Keep it short (5–12 words) and avoid duplicating words. "
        f"Preserve the following nouns exactly as they are: {nouns_str}.\n{clean_title}"
    )

    encoding = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOTAL_TOKENS
    ).to(device)

    available_output_tokens = 60
    target_word_count = len(clean_title.split())
    min_wc = max(1, int(target_word_count * 0.6))
    max_wc = min(12, int(target_word_count * 1.4))

    best_paraphrase = None
    best_score = -1
    best_attempt = ""
    best_metadata = ""

    for attempt in range(max_attempts):
        sub_attempt = 0
        valid_paraphrase_found = False

        while not valid_paraphrase_found and sub_attempt < max_sub_attempts:
            try:
                with torch.no_grad():
                    output = model.generate(
                        input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask'],
                        max_new_tokens=available_output_tokens,
                        do_sample=True,
                        top_k=40,
                        top_p=0.95,
                        temperature=0.8 + 0.1 * sub_attempt,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        num_return_sequences=MAX_RETURN_SEQUENCES
                    )

                decoded_outputs = [
                    tokenizer.decode(seq, skip_special_tokens=True).strip()
                    for seq in output
                ]

                for idx, d in enumerate(decoded_outputs):
                    paraphrased = d.replace(prompt, "").strip() if prompt in d else d.strip()
                    paraphrased = clean_description(paraphrased)
                    paraphrased = restore_capitalization(paraphrased, capitalized_words)

                    if not paraphrased or len(paraphrased.split()) < 1:
                        logger.info(f"⛔ Rejected due to empty or too short: \"{paraphrased}\"")
                        continue

                    is_banned, banned_phrase, context_snippet = contains_banned_phrase(paraphrased, [])
                    if is_banned:
                        logger.info(f"⛔ Rejected due to banned phrase '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\"")
                        print(f"⛔ Rejected due to banned phrase '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\"")
                        continue
                    if has_repetitions(paraphrased):
                        logger.info(f"⛔ Rejected due to repeated phrases: \"{paraphrased}\"")
                        continue
                    if not is_grammatically_correct(paraphrased):
                        logger.info(f"⛔ Rejected due to grammar: \"{paraphrased}\"")
                        continue
                    if not contains_nouns(paraphrased, nouns):
                        logger.info(f"⛔ Rejected due to missing nouns: \"{paraphrased}\" (required: {nouns})")
                        continue

                    score, sim, wc = score_paraphrase(title, paraphrased, target_word_count)
                    first_diff = not paraphrased.lower().startswith(title.lower())

                    print(f"📝 Attempt {attempt + 1}.{sub_attempt + 1}, Option {idx + 1}")
                    print(f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}")
                    print(f"→ Paraphrased: {paraphrased}\n")

                    is_valid = (
                        min_wc <= wc <= max_wc
                        and sim >= (0.65 if target_word_count > 5 else 0.6)
                        and first_diff
                    )

                    if is_valid:
                        print(f"✅ Picked from attempt {attempt + 1}.{sub_attempt + 1}, option {idx + 1}")
                        print(f"→ {paraphrased}\n")
                        return paraphrased

                    if first_diff and score > best_score:
                        best_score = score
                        best_paraphrase = paraphrased
                        best_attempt = f"{attempt + 1}.{sub_attempt + 1}, option {idx + 1}"
                        best_metadata = (
                            f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}\n"
                            f"→ Paraphrased: {paraphrased}"
                        )

                sub_attempt += 1
                time.sleep(0.5 * (2 ** sub_attempt))

            except Exception as e:
                logger.error(f"Error during attempt {attempt + 1}, sub-attempt {sub_attempt + 1}: {str(e)}")
                sub_attempt += 1
                time.sleep(0.5 * (2 ** sub_attempt))

        time.sleep(1)

    if best_paraphrase:
        print(f"✅ Picked fallback from attempt {best_attempt}")
        print(best_metadata + "\n")
        return best_paraphrase

    print("❌ Fallback to original title.\n")
    return clean_title

def paraphrase_strict_company(text, max_attempts=2, max_sub_attempts=2):
    def contains_prompt(para):
        prompt_phrases = [
            "Rephrase the following company details paragraph",
            "Rephrase the company details",
            "Paragraph professionally, preserving all key details",
            "Rewrite the following",
            "Rephrase the paragraph below",
            "Rephrase the following company details",
            "Preserving all key details",
            "Tone and structure",
            "Keep the length approximately the same",
            "Do your company information paragraph need improvements",
            "Paraphrase", "Paraphrased", "Paraphrasing", "Paragraph", "Company details",
        ]
        para_lower = para.lower()
        for phrase in prompt_phrases:
            if phrase.lower() in para_lower:
                start_idx = para_lower.find(phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(para), start_idx + len(phrase) + 20)
                context_snippet = para[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(para):
                    context_snippet = context_snippet + "..."
                return True, phrase, context_snippet
        return False, None, None

    clean_text = sanitize_text(text)
    if not clean_text:
        logger.error("Input text is empty after sanitization.")
        return text

    capitalized_words = extract_capitalized_words(clean_text)
    logger.debug(f"Extracted capitalized words from text: {list(capitalized_words.values())}")

    paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
    final_paraphrased = []

    for idx, para in enumerate(paragraphs):
        print(f"\n🔹 Paraphrasing Paragraph {idx + 1}/{len(paragraphs)}")

        prompt = (
            f"Rephrase the following company details paragraph professionally, preserving all key details, tone, and structure. "
            f"Keep the length approximately the same and avoid repeating the input format:\n{para}"
        )

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_token_len = len(prompt_tokens)

        if prompt_token_len > MAX_TOTAL_TOKENS - 200:
            logger.warning(f"Prompt for paragraph {idx + 1} too long, truncating to fit.")
            para = " ".join(para.split()[:int((MAX_TOTAL_TOKENS - 200) / 4)])
            prompt = (
                f"Rephrase the following company details paragraph professionally, preserving all key details, tone, and structure. "
                f"Keep the length approximately the same:\n{para}"
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_token_len = len(prompt_tokens)

        available_output_tokens = max(200, MAX_TOTAL_TOKENS - prompt_token_len)
        target_word_count = len(para.split())
        tolerance = 0.25
        min_wc = int(target_word_count * (1 - tolerance))
        max_wc = int(target_word_count * (1 + tolerance))

        encoding = tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOTAL_TOKENS
        ).to(device)

        best_paraphrase = None
        best_score = -1
        best_attempt = ""
        best_metadata = ""

        for attempt in range(max_attempts):
            sub_attempt = 0
            valid_paraphrase_found = False

            while not valid_paraphrase_found and sub_attempt < max_sub_attempts:
                try:
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'],
                            max_new_tokens=available_output_tokens,
                            do_sample=True,
                            top_k=40,
                            top_p=0.95,
                            temperature=0.9 + 0.1 * sub_attempt,
                            repetition_penalty=1.1,
                            no_repeat_ngram_size=2,
                            num_return_sequences=MAX_RETURN_SEQUENCES
                        )

                    decoded = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in output]

                    for option_index, d in enumerate(decoded):
                        paraphrased = d.replace(prompt, "").strip() if prompt in d else d.strip()
                        paraphrased = clean_description(paraphrased)
                        paraphrased = restore_capitalization(paraphrased, capitalized_words)

                        if not paraphrased or len(paraphrased.split()) < 5:
                            logger.info(f"⛔ Rejected due to empty or too short: \"{paraphrased}\"")
                            continue
                        is_banned, banned_phrase, context_snippet = contains_prompt(paraphrased)
                        if is_banned:
                            print(f"❌ Rejected due to prompt echo (phrase: '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\")")
                            logger.info(f"❌ Rejected due to prompt echo (phrase: '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\")")
                            continue

                        word_count = len(paraphrased.split())
                        similarity = is_good_paraphrase(para, paraphrased)
                        score = (similarity + (1 - abs(word_count - target_word_count) / max(target_word_count, 1))) / 2

                        first_sentence = paraphrased.split(".")[0].strip()
                        original_first = para.split(".")[0].strip()
                        first_diff = not first_sentence.lower().startswith(original_first.lower())

                        print(f"📝 Attempt {attempt + 1}.{sub_attempt + 1}, Option {option_index + 1}")
                        print(f"↪ Words: {word_count}, Sim: {similarity:.2f}, Score: {score:.2f}, First sentence different: {first_diff}")
                        print(f"→ First sentence: {first_sentence}\n")

                        is_valid = (
                            min_wc <= word_count <= max_wc
                            and similarity >= (0.65 if target_word_count > 10 else 0.6)
                            and first_diff
                            and is_grammatically_correct(paraphrased)
                        )

                        if is_valid:
                            print(f"✅ Picked from attempt {attempt + 1}.{sub_attempt + 1}, option {option_index + 1}")
                            final_paraphrased.append(clean_description(paraphrased))
                            valid_paraphrase_found = True
                            break

                        if first_diff and score > best_score:
                            best_score = score
                            best_paraphrase = paraphrased
                            best_attempt = f"{attempt + 1}.{sub_attempt + 1}, option {option_index + 1}"
                            best_metadata = (
                                f"↪ Words: {word_count}, Sim: {similarity:.2f}, Score: {score:.2f}, First sentence different: {first_diff}\n"
                                f"→ First sentence: {first_sentence}"
                            )

                    if not valid_paraphrase_found:
                        sub_attempt += 1
                        time.sleep(0.5 * (2 ** sub_attempt))

                except Exception as e:
                    logger.error(f"Error during attempt {attempt + 1}, sub-attempt {sub_attempt + 1} for paragraph {idx + 1}: {str(e)}")
                    sub_attempt += 1
                    time.sleep(0.5 * (2 ** sub_attempt))

            if valid_paraphrase_found:
                break
            time.sleep(1)

        if not valid_paraphrase_found:
            if best_paraphrase:
                print(f"✅ Picked fallback from attempt {best_attempt}")
                print(best_metadata + "\n")
                final_paraphrased.append(clean_description(best_paraphrase))
            else:
                print(f"❌ Paragraph {idx + 1} fallback to original.\n")
                final_paraphrased.append(para)

    return "\n\n".join(final_paraphrased)

def paraphrase_strict_tagline(company_tagline, max_attempts=5):
    clean_text = sanitize_text(company_tagline)
    if not clean_text:
        logger.error(f"Input text is empty after sanitization: {company_tagline}")
        print("Error: Input text is empty after sanitization.")
        return company_tagline

    capitalized_words = extract_capitalized_words(clean_text)
    logger.debug(f"Extracted capitalized words from tagline: {list(capitalized_words.values())}")

    target_word_count = max(len(clean_text.split()), 8)
    min_word_count = 4
    max_word_count = 15

    rejected_phrases = [
        "Paraphrased tagline", "Rewrite the following", "Original tagline",
        "Professionally rewritten", "Crisp and impactful", "Summary:",
        "Short and professional", "Keep it short", "###", "Tagline:",
        "Output:", "Company summary", "Paraphrased version", "Rephrased version",
        "Paraphrase", "Paraphrased", "Paraphrasing", "Summarized", "Summarised",
        "Summarizing", "Summarising", "Summary"
    ]

    def contains_rejected_phrase(text):
        lower = text.lower()
        for bad_phrase in rejected_phrases:
            if bad_phrase.lower() in lower:
                start_idx = lower.find(bad_phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(text), start_idx + len(bad_phrase) + 20)
                context_snippet = text[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(text):
                    context_snippet = context_snippet + "..."
                return True, bad_phrase, context_snippet
        return False, None, None

    def first_sentence_diff(original, paraphrased):
        orig_first = original.split(".")[0].strip().lower()
        para_first = paraphrased.split(".")[0].strip().lower()
        return not para_first.startswith(orig_first)

    input_prompt = (
        f"Rewrite the following tagline into a crisp, professional, and meaningful summary. "
        f"Keep it short and impactful (5–12 words):\n\n"
        f"### Original ###\n{clean_text}\n\n### Paraphrased Tagline ###"
    )

    input_tokens = tokenizer.encode(input_prompt, add_special_tokens=True)
    max_length = min(len(input_tokens) + 50, 512)

    encoding = tokenizer.encode_plus(
        input_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    best_paraphrase = None
    best_score = -1
    best_meta = {"attempt": -1, "similarity": 0.0, "word_count": 0, "first_diff": False}

    for attempt in range(max_attempts):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    max_new_tokens=25,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    num_return_sequences=6,
                    eos_token_id=tokenizer.eos_token_id
                )

            decoded_outputs = [
                tokenizer.decode(seq, skip_special_tokens=True).strip()
                for seq in outputs
            ]

            paraphrases = []
            for d in decoded_outputs:
                if "### Paraphrased Tagline ###" in d:
                    parts = d.split("### Paraphrased Tagline ###")
                    paraphrased = clean_description(parts[1]) if len(parts) >= 2 else clean_description(d)
                else:
                    paraphrased = clean_description(d)
                paraphrased = restore_capitalization(paraphrased, capitalized_words)
                paraphrases.append(paraphrased)

            for paraphrased in paraphrases:
                is_banned, banned_phrase, context_snippet = contains_rejected_phrase(paraphrased)
                if is_banned:
                    logger.info(f"⛔ Rejected tagline due to banned phrase '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\"")
                    print(f"⛔ Rejected tagline due to banned phrase '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\"")
                    continue

                if not is_grammatically_correct(paraphrased):
                    logger.info(f"⛔ Rejected tagline due to grammar: \"{paraphrased}\"")
                    continue

                word_count = len(paraphrased.split())
                if word_count < min_word_count or word_count > max_word_count:
                    continue

                similarity = is_good_paraphrase(clean_text, paraphrased)
                length_score = 1 - abs(target_word_count - word_count) / target_word_count
                score = similarity * 0.7 + length_score * 0.3
                first_diff = first_sentence_diff(clean_text, paraphrased)

                print(f"Attempt {attempt + 1}: \"{paraphrased}\" | Words: {word_count} | Similarity: {similarity:.2f} | Score: {score:.2f} | First sentence different: {first_diff}")

                if first_diff and score > best_score:
                    best_paraphrase = paraphrased
                    best_meta = {
                        "attempt": attempt + 1,
                        "similarity": similarity,
                        "word_count": word_count,
                        "first_diff": first_diff
                    }

        except Exception as e:
            logger.error(f"Error during paraphrasing attempt {attempt + 1}: {str(e)}")

        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)

    if best_paraphrase:
        logger.info(
            f"✅ Picked tagline from attempt {best_meta['attempt']} "
            f"(words: {best_meta['word_count']}, similarity: {best_meta['similarity']:.2f}, score: {best_score:.2f}, first sentence different: {best_meta['first_diff']})"
        )
        print(
            f"\n✅ Picked tagline from attempt {best_meta['attempt']} "
            f"(words: {best_meta['word_count']}, similarity: {best_meta['similarity']:.2f}, score: {best_score:.2f}, first sentence different: {best_meta['first_diff']})"
        )
        return best_paraphrase

    logger.warning("No valid tagline candidates produced. Returning original.")
    return clean_text

def paraphrase_strict_description(text, max_attempts=2, max_sub_attempts=2):
    def contains_prompt(para):
        prompt_phrases = [
            "Rephrase the following job description paragraph",
            "Rephrase the job description",
            "Paragraph professionally, preserving all key details",
            "Rewrite the following",
            "Rephrase the paragraph below",
            "Rephrase the following job description",
            "Preserving all key details",
            "Tone and structure",
            "Keep the length approximately the same",
            "Job description paragraph professionally",
            "Paraphrase", "Paraphrased", "Paraphrase the following",
            "Paraphrase the job description", "Paraphrasing",
            "Job description", "Job description paragraph",
        ]
        para_lower = para.lower()
        for phrase in prompt_phrases:
            if phrase.lower() in para_lower:
                start_idx = para_lower.find(phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(para), start_idx + len(phrase) + 20)
                context_snippet = para[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(para):
                    context_snippet = context_snippet + "..."
                return True, phrase, context_snippet
        return False, None, None

    clean_text = sanitize_text(text)
    if not clean_text:
        logger.error("Input text is empty after sanitization.")
        return text

    capitalized_words = extract_capitalized_words(clean_text)
    logger.debug(f"Extracted capitalized words from text: {list(capitalized_words.values())}")

    paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
    final_paraphrased = []

    for idx, para in enumerate(paragraphs):
        print(f"\n🔹 Paraphrasing Paragraph {idx + 1}/{len(paragraphs)}")

        prompt = (
            f"Rephrase the following job description paragraph professionally, preserving all key details, tone, and structure. "
            f"Keep the length approximately the same and avoid repeating the input format:\n{para}"
        )

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_token_len = len(prompt_tokens)

        if prompt_token_len > MAX_TOTAL_TOKENS - 200:
            logger.warning(f"Prompt for paragraph {idx + 1} too long, truncating to fit.")
            para = " ".join(para.split()[:int((MAX_TOTAL_TOKENS - 200) / 4)])
            prompt = (
                f"Rephrase the following job description paragraph professionally, preserving all key details, tone, and structure. "
                f"Keep the length approximately the same:\n{para}"
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_token_len = len(prompt_tokens)

        available_output_tokens = max(200, MAX_TOTAL_TOKENS - prompt_token_len)
        target_word_count = len(para.split())
        tolerance = 0.25
        min_wc = int(target_word_count * (1 - tolerance))
        max_wc = int(target_word_count * (1 + tolerance))

        encoding = tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOTAL_TOKENS
        ).to(device)

        best_paraphrase = None
        best_score = -1
        best_attempt = ""
        best_metadata = ""

        for attempt in range(max_attempts):
            sub_attempt = 0
            valid_paraphrase_found = False

            while not valid_paraphrase_found and sub_attempt < max_sub_attempts:
                try:
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'],
                            max_new_tokens=available_output_tokens,
                            do_sample=True,
                            top_k=40,
                            top_p=0.95,
                            temperature=0.9 + 0.1 * sub_attempt,
                            repetition_penalty=1.1,
                            no_repeat_ngram_size=2,
                            num_return_sequences=MAX_RETURN_SEQUENCES
                        )

                    decoded = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in output]

                    for option_index, d in enumerate(decoded):
                        paraphrased = d.replace(prompt, "").strip() if prompt in d else d.strip()
                        paraphrased = clean_description(paraphrased)
                        paraphrased = restore_capitalization(paraphrased, capitalized_words)

                        if not paraphrased or len(paraphrased.split()) < 5:
                            logger.info(f"⛔ Rejected due to empty or too short: \"{paraphrased}\"")
                            continue
                        is_banned, banned_phrase, context_snippet = contains_prompt(paraphrased)
                        if is_banned:
                            print(f"❌ Rejected due to prompt echo (phrase: '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\")")
                            logger.info(f"❌ Rejected due to prompt echo (phrase: '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\")")
                            continue

                        word_count = len(paraphrased.split())
                        similarity = is_good_paraphrase(para, paraphrased)
                        score = (similarity + (1 - abs(word_count - target_word_count) / max(target_word_count, 1))) / 2

                        first_sentence = paraphrased.split(".")[0].strip()
                        original_first = para.split(".")[0].strip()
                        first_diff = not first_sentence.lower().startswith(original_first.lower())

                        print(f"📝 Attempt {attempt + 1}.{sub_attempt + 1}, Option {option_index + 1}")
                        print(f"↪ Words: {word_count}, Sim: {similarity:.2f}, Score: {score:.2f}, First sentence different: {first_diff}")
                        print(f"→ First sentence: {first_sentence}\n")

                        is_valid = (
                            min_wc <= word_count <= max_wc
                            and similarity >= (0.65 if target_word_count > 10 else 0.6)
                            and first_diff
                            and is_grammatically_correct(paraphrased)
                        )

                        if is_valid:
                            print(f"✅ Picked from attempt {attempt + 1}.{sub_attempt + 1}, option {option_index + 1}")
                            final_paraphrased.append(clean_description(paraphrased))
                            valid_paraphrase_found = True
                            break

                        if first_diff and score > best_score:
                            best_score = score
                            best_paraphrase = paraphrased
                            best_attempt = f"{attempt + 1}.{sub_attempt + 1}, option {option_index + 1}"
                            best_metadata = (
                                f"↪ Words: {word_count}, Sim: {similarity:.2f}, Score: {score:.2f}, First sentence different: {first_diff}\n"
                                f"→ First sentence: {first_sentence}"
                            )

                    if not valid_paraphrase_found:
                        sub_attempt += 1
                        time.sleep(0.5 * (2 ** sub_attempt))

                except Exception as e:
                    logger.error(f"Error during attempt {attempt + 1}, sub-attempt {sub_attempt + 1} for paragraph {idx + 1}: {str(e)}")
                    sub_attempt += 1
                    time.sleep(0.5 * (2 ** sub_attempt))

            if valid_paraphrase_found:
                break
            time.sleep(1)

        if not valid_paraphrase_found:
            if best_paraphrase:
                print(f"✅ Picked fallback from attempt {best_attempt}")
                print(best_metadata + "\n")
                final_paraphrased.append(clean_description(best_paraphrase))
            else:
                print(f"❌ Paragraph {idx + 1} fallback to original.\n")
                final_paraphrased.append(para)

    return "\n\n".join(final_paraphrased)

def load_mauritius_processed_job_ids():
    if not os.path.exists(PROCESSED_IDS_FILE):
        logger.info(f"{PROCESSED_IDS_FILE} does not exist. Initializing empty sets.")
        return set(), set(), set()
    try:
        df = pd.read_csv(PROCESSED_IDS_FILE)
        required_columns = ['Job ID', 'Job URL', 'Company Name', 'URL Page', 'Job Number']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        job_ids = set(df['Job ID'].fillna('').astype(str).tolist())
        job_urls = set(df['Job URL'].fillna('').astype(str).tolist())
        company_names = set(df['Company Name'].fillna('').astype(str).tolist())
        logger.info(f"Loaded {len(job_ids)} Job IDs, {len(job_urls)} Job URLs, and {len(company_names)} Company Names from {PROCESSED_IDS_FILE}")
        return job_ids, job_urls, company_names
    except Exception as e:
        logger.error(f"Error reading {PROCESSED_IDS_FILE}: {str(e)}. Initializing empty sets.")
        print(f"Error reading {PROCESSED_IDS_FILE}: {str(e)}. Using empty sets.")
        return set(), set(), set()

def save_processed_job_id(job_id, job_url, company_name, url_page, job_number):
    try:
        job_id = str(job_id)
        job_url = sanitize_text(str(job_url), is_url=True)
        company_name = sanitize_text(str(company_name))
        url_page = str(url_page)
        job_number = str(job_number)
        new_row = pd.DataFrame({
            'Job ID': [job_id],
            'Job URL': [job_url],
            'Company Name': [company_name],
            'URL Page': [url_page],
            'Job Number': [job_number]
        })
        if os.path.exists(PROCESSED_IDS_FILE):
            df = pd.read_csv(PROCESSED_IDS_FILE)
            if not df.empty and job_id not in df['Job ID'].astype(str).values:
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(PROCESSED_IDS_FILE, index=False)
            elif df.empty:
                new_row.to_csv(PROCESSED_IDS_FILE, index=False)
        else:
            new_row.to_csv(PROCESSED_IDS_FILE, index=False)
        logger.info(f"Saved Job ID {job_id}, URL {job_url}, Company {company_name}, Page {url_page}, Job Number {job_number} to {PROCESSED_IDS_FILE}")
    except Exception as e:
        logger.error(f"Error saving Job ID {job_id}: {str(e)}")
        print(f"Error saving Job ID {job_id}: {str(e)}")

def load_last_processed_page():
    try:
        if os.path.exists(LAST_PAGE_FILE):
            with open(LAST_PAGE_FILE, 'r') as f:
                page = int(f.read().strip())
                logger.info(f"Loaded last processed page: {page}")
                return page
        logger.info("No last processed page file found. Starting from page 1.")
        return 1
    except Exception as e:
        logger.error(f"Error loading last processed page: {str(e)}")
        print(f"Error loading last processed page: {str(e)}. Starting from page 1.")
        return 1

def save_last_processed_page(page):
    try:
        with open(LAST_PAGE_FILE, 'w') as f:
            f.write(str(page))
        logger.info(f"Saved last processed page: {page}")
    except Exception as e:
        logger.error(f"Error saving last processed page {page}: {str(e)}")
        print(f"Error saving last processed page {page}: {str(e)}")

def validate_application_method(value, is_email=False):
    if not value:
        return False
    if is_email:
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value))
    return bool(re.match(r'^https?://[^\s/$.?#].[^\s]*$', value))

def clean_application_url(url):
    if not url:
        return url
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        response.raise_for_status()
        final_url = response.url
        parsed_url = urlparse(final_url)
        query_params = parse_qs(parsed_url.query)
        query_params.pop('utm_source', None)
        cleaned_query = urlencode(query_params, doseq=True)
        cleaned_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            cleaned_query,
            parsed_url.fragment
        ))
        logger.debug(f"Cleaned application URL: {final_url} -> {cleaned_url}")
        return cleaned_url
    except Exception as e:
        logger.error(f"Error processing application URL {url}: {str(e)}")
        return url

def upload_logo_to_media_library(logo_url, auth, headers):
    if not logo_url or not logo_url.startswith('http') or not (logo_url.lower().endswith('.png') or logo_url.lower().endswith('.jpg') or logo_url.lower().endswith('.jpeg')):
        logger.warning(f"Invalid logo URL or format: {logo_url}")
        return None
    try:
        response = requests.get(logo_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('content-type', 'image/jpeg')
        filename = logo_url.split('/')[-1] or 'company_logo.jpg'
        media_headers = headers.copy()
        media_headers['Content-Disposition'] = f'attachment; filename={filename}'
        media_headers['Content-Type'] = content_type
        response = requests.post(WP_MEDIA_URL, headers=media_headers, data=response.content, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=10, verify=False)
        response.raise_for_status()
        media = response.json()
        attachment_id = media.get('id')
        logger.info(f"Uploaded logo {logo_url} to media library, Attachment ID: {attachment_id}")
        return attachment_id
    except RequestException as e:
        logger.error(f"Error uploading logo {logo_url}: {str(e)}")
        return None

def extract_job_title(job_title):
    patterns = [
        (r'The most accurate and professional paraphrase would be:\s*(["“])?([^”"\n]+)(["”])?', 2),
        (r'paraphrase would be:\s*(["“])?([^”"\n]+)(["”])?', 2),
        (r'["“]([^”"]+)["”]\s*This paraphrase maintains the original meaning', 1),
        (r'A good paraphrase would be:\s*["“]([^”"]+)["”]', 1),
        (r'["“]([^”]+)["”]', 1),
        (r'–\s*([^-]+)\s*–', 1),
        (r'Here’s a :\s*(.+)', 1),
        (r'\* Job title:\s*([^\n*]+)', 1)
    ]
    for pattern, group in patterns:
        match = re.search(pattern, job_title, re.IGNORECASE)
        if match:
            extracted = match.group(group).strip()
            words = extracted.split()
            if len(words) > 30:
                words = words[:30]
                while words and words[-1].lower() in {'and', 'or', 'at', 'in', 'of', 'for', 'with'}:
                    words.pop()
                extracted = ' '.join(words)
            logger.debug(f"Matched pattern '{pattern}' for job title, extracted: {extracted}")
            return extracted
    extracted = job_title.strip()
    words = extracted.split()
    if len(words) > 30:
        words = words[:30]
        while words and words[-1].lower() in {'and', 'or', 'at', 'in', 'of', 'for', 'with'}:
            words.pop()
        extracted = ' '.join(words)
    logger.debug(f"No pattern matched, using fallback extracted title: {extracted}")
    return extracted

def print_word_by_word(text, delay=0.05):
    text = re.sub(r'\*\*', '', text)
    print("\nStep 3: API Response (Word-by-Word)")
    print("-" * 30)
    print("Rewritten Text: ")
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        words = para.split()
        for word in words:
            print(f"{word} ", end="", flush=True)
            time.sleep(delay)
        print('\n')
    return text

def paraphrase_title_and_description(title, description, index, max_attempts=5):
    print(f"\n=== Processing Article #{index + 1} ===")
    print("Step 1: Original Article Text")
    print("-" * 30)
    print(f"Original Job Title: {title}")
    print(f"Original Job Description: {description}")

    print("\nStep 2: Paraphrasing Title and Description Separately")
    print("-" * 30)

    try:
        print(f"Paraphrasing Job Title: {title}")
        paraphrased_title = paraphrase_strict_title(title, max_attempts=max_attempts)
        logger.debug(f"Raw paraphrased title: {paraphrased_title}")
        print(f"Paraphrased Job Title: {paraphrased_title}")

        words = paraphrased_title.split()
        if len(words) > 30:
            words = words[:30]
            while words and words[-1].lower() in {'and', 'or', 'at', 'in', 'of', 'for', 'with'}:
                words.pop()
            paraphrased_title = ' '.join(words)
            logger.debug(f"Truncated paraphrased title: {paraphrased_title}")

        rewritten_title = paraphrased_title

    except Exception as e:
        logger.error(f"Error paraphrasing title: {str(e)}. Falling back to original title.")
        print(f"Error paraphrasing title: {str(e)}. Falling back to original title.")
        rewritten_title = title

    try:
        print(f"Paraphrasing Job Description: {description}")
        paraphrased_description = paraphrase_strict_description(description, max_attempts=max_attempts)
        logger.debug(f"Raw paraphrased description: {paraphrased_description}")
        print(f"Paraphrased Job Description: {paraphrased_description}")
        rewritten_description = clean_description(paraphrased_description)

    except Exception as e:
        logger.error(f"Error paraphrasing description: {str(e)}. Falling back to original description.")
        print(f"Error paraphrasing description: {str(e)}. Falling back to original description.")
        rewritten_description = description

    print(f"\nStep 3: Final Paraphrased Output")
    print("-" * 30)
    print(f"Extracted Paraphrased Job Title: {rewritten_title}")
    print(f"Extracted Paraphrased Job Description: {rewritten_description}")

    return print_word_by_word(f"Job Title: {rewritten_title}\n\nJob Description:\n{rewritten_description}"), rewritten_title, rewritten_description

def get_region_term_id(location_value, auth, headers):
    taxonomy_url = "https://mauritius.mimusjobs.com/wp-json/wp/v2/job_listing_region"
    location_slug = location_value.lower().replace(' ', '-')
    try:
        response = requests.get(f"{taxonomy_url}?slug={location_slug}", headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        terms = response.json()
        if terms:
            logger.debug(f"Found existing region term: {terms[0]['id']} for {location_value}")
            return terms[0]['id']
    except RequestException as e:
        logger.error(f"Error fetching region term for {location_value}: {str(e)}")
    try:
        term_data = {"name": location_value, "slug": location_slug}
        response = requests.post(taxonomy_url, json=term_data, headers=headers, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=10, verify=False)
        response.raise_for_status()
        term = response.json()
        logger.debug(f"Created new region term: {term['id']} for {location_value}")
        return term['id']
    except RequestException as e:
        logger.error(f"Error creating region term for {location_value}: {str(e)}")
        return None

def get_job_type_term_id(job_type_value, auth, headers):
    taxonomy_url = "https://mauritius.mimusjobs.com/wp-json/wp/v2/job_listing_type"
    job_type_slug = job_type_value.lower().replace(' ', '-')
    try:
        response = requests.get(f"{taxonomy_url}?slug={job_type_slug}", headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        terms = response.json()
        if terms:
            logger.debug(f"Found existing job type term: {terms[0]['id']} for {job_type_value}")
            return terms[0]['id']
    except RequestException as e:
        logger.error(f"Error fetching job type term for {job_type_value}: {str(e)}")
    try:
        term_data = {"name": job_type_value, "slug": job_type_slug}
        response = requests.post(taxonomy_url, json=term_data, headers=headers, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=10, verify=False)
        response.raise_for_status()
        term = response.json()
        logger.debug(f"Created new job type term: {term['id']} for {job_type_value}")
        return term['id']
    except RequestException as e:
        logger.error(f"Error creating job type term for {job_type_value}: {str(e)}")
        return None

def initialize_job_type_terms(auth, headers):
    taxonomy_url = "https://mauritius.mimusjobs.com/wp-json/wp/v2/job_listing_type"
    for job_type, slug in JOB_TYPE_MAPPING.items():
        try:
            response = requests.get(f"{taxonomy_url}?slug={slug}", headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            terms = response.json()
            if not terms:
                term_data = {"name": job_type, "slug": slug}
                response = requests.post(taxonomy_url, json=term_data, headers=headers, auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=10, verify=False)
                response.raise_for_status()
                term = response.json()
                logger.info(f"Initialized job type term: {term['id']} for {job_type}")
        except RequestException as e:
            logger.error(f"Error initializing job type term {job_type}: {str(e)}")

def save_company_to_wordpress(index, company_data):
    auth_string = f"{WP_USERNAME}:{WP_APP_PASSWORD}"
    auth = base64.b64encode(auth_string.encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    company_name = sanitize_text(company_data.get("company_name", "Unknown Company"))
    logo_url = sanitize_text(company_data.get("company_logo", []), is_url=True)
    logo_url = logo_url[0] if isinstance(logo_url, list) and logo_url else ""
    attachment_id = None
    if logo_url:
        attachment_id = upload_logo_to_media_library(logo_url, auth, headers)
    else:
        logger.info(f"No valid logo URL for company {company_name}. Skipping logo upload.")
    company_details = company_data.get("company_details", "")
    if company_details:
        print(f"\nParaphrasing Company Details for {company_name}")
        print("-" * 30)
        print(f"Original Company Details: {company_details}")
        paraphrased_details = paraphrase_strict_company(company_details, max_attempts=5)
        paraphrased_details = re.sub(r'Job Title:\s*[^\n]*\n*', '', paraphrased_details, flags=re.IGNORECASE)
        paraphrased_details = re.sub(r'Job Description:\s*', '', paraphrased_details, flags=re.IGNORECASE)
        sentences = nltk.sent_tokenize(paraphrased_details)
        paragraphs = []
        current_paragraph = []
        sentence_count = 0
        for sentence in sentences:
            current_paragraph.append(sentence)
            sentence_count += 1
            if sentence_count >= 3:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
                sentence_count = 0
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        paraphrased_details = '\n\n'.join(paragraphs)
        print(f"Paraphrased Company Details: {paraphrased_details}")
        company_details = clean_description(paraphrased_details)
    else:
        logger.warning(f"No company details to paraphrase for {company_name}")
        company_details = ""
    company_tagline = sanitize_text(company_data.get("company_details", ""))
    if company_tagline:
        print(f"\nParaphrasing Company Tagline for {company_name}")
        print("-" * 30)
        print(f"Original Company Tagline: {company_tagline}")
        paraphrased_tagline = paraphrase_strict_tagline(company_tagline, max_attempts=5)
        paraphrased_tagline = re.sub(r'Job Title:\s*[^\n]*\n*', '', paraphrased_tagline, flags=re.IGNORECASE)
        paraphrased_tagline = re.sub(r'Job Description:\s*', '', paraphrased_tagline, flags=re.IGNORECASE)
        print(f"Paraphrased Company Tagline: {paraphrased_tagline}")
        company_tagline = clean_description(paraphrased_tagline)
    else:
        logger.warning(f"No company tagline to paraphrase for {company_name}")
        company_tagline = ""
    check_url = f"{WP_COMPANY_URL}?slug={company_name.lower().replace(' ', '-')}"
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.get(check_url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        posts = response.json()
        if posts:
            post = posts[0]
            logger.info(f"Company {company_name} already exists: Post ID {post.get('id')}, URL {post.get('link')}")
            return post.get("id"), post.get("link")
    except RequestException as e:
        logger.warning(f"Error checking for existing company {company_name}: {str(e)}. Proceeding to create new company.")
    post_data = {
        "title": company_name,
        "content": company_details,
        "status": "publish",
        "featured_media": attachment_id if attachment_id else 0,
        "meta": {
            "_company_name": company_name,
            "_company_logo": str(attachment_id) if attachment_id else "",
            "_company_industry": sanitize_text(company_data.get("company_industry", "")),
            "_company_founded": sanitize_text(company_data.get("company_founded", "")),
            "_company_type": sanitize_text(company_data.get("company_type", "")),
            "_company_website": sanitize_text(company_data.get("company_website", ""), is_url=True),
            "_company_address": sanitize_text(company_data.get("company_address", "")),
            "_company_tagline": company_tagline
        }
    }
    logger.debug(f"Sending company payload to WordPress for company {company_name}: {json.dumps(post_data, indent=2)}")
    try:
        response = session.post(WP_COMPANY_URL, json=post_data, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        post = response.json()
        logger.info(f"Successfully posted company {company_name} to WordPress: Post ID {post.get('id')}, URL {post.get('link')}")
        print(f"\nStep 5: Published Company to WordPress")
        print("-" * 30)
        print(f"Company Post ID: {post.get('id')}")
        print(f"Company Post URL: {post.get('link')}")
        return post.get("id"), post.get("link")
    except RequestException as e:
        logger.error(f"Failed to post company {company_name}: {str(e)}")
        print(f"Error publishing company {company_name}: {e}")
        return None, None

def save_article_to_wordpress(index, job_data, rewritten_title, rewritten_description, application):
    auth_string = f"{WP_USERNAME}:{WP_APP_PASSWORD}"
    auth = base64.b64encode(auth_string.encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json"
    }
    initialize_job_type_terms(auth, headers)
    location_value = sanitize_text(job_data.get("Location", "Remote"))
    job_type_value = sanitize_text(job_data.get("Job Type", "Full-time"))
    job_type_slug = JOB_TYPE_MAPPING.get(job_type_value, "full-time").lower()
    logo_url = sanitize_text(job_data.get("Company Logo", ""), is_url=True)
    company_name = sanitize_text(job_data.get("Company", "Unknown Company"))
    job_id = str(job_data.get("Job ID", ""))
    job_url = sanitize_text(job_data.get("Job URL", ""), is_url=True)
    is_email = validate_application_method(application, is_email=True)
    is_url = validate_application_method(application, is_email=False)
    if not (is_email or is_url):
        logger.warning(f"Invalid application method for job {index + 1} (Job ID: {job_id}): {application}. Setting to empty.")
        application = ""
    title_slug = rewritten_title.lower().replace(' ', '-') if rewritten_title and not rewritten_title.startswith("Error:") else sanitize_text(job_data.get("Job Title", f"job-listing-{index + 1}")).lower().replace(' ', '-')
    check_url = f"{WP_URL}?slug={title_slug}"
    session = requests.Session()
    retries = Retry(total=0, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.get(check_url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        posts = response.json()
        if posts:
            post = posts[0]
            logger.info(f"Job {index + 1} (Job ID: {job_id}) already exists in WordPress: Post ID {post.get('id')}, URL {post.get('link')}")
            print(f"Skipping job {index + 1}: Already exists in WordPress with Post ID {post.get('id')}, URL {post.get('link')}")
            save_processed_job_id(job_id, job_url, company_name, job_data.get('URL Page', ''), job_data.get('Job Number', ''))
            return post.get("id"), post.get("link")
    except RequestException as e:
        logger.warning(f"Error checking for existing job {index + 1}: {str(e)}. Proceeding to create new post.")
    attachment_id = upload_logo_to_media_library(logo_url, auth, headers)
    region_term_id = get_region_term_id(location_value, auth, headers)
    job_type_term_id = get_job_type_term_id(job_type_value, auth, headers)
    if company_name == "Unknown Company":
        logger.warning(f"Using fallback company name 'Unknown Company' for job {index + 1}")
    post_data = {
        "title": rewritten_title if rewritten_title and not rewritten_title.startswith("Error:") else sanitize_text(job_data.get("Job Title", f"Job Listing {index + 1}")),
        "content": rewritten_description if rewritten_description and not rewritten_description.startswith("Error:") else sanitize_text(job_data.get("Job Description", "")),
        "status": "publish",
        "featured_media": attachment_id if attachment_id else 0,
        "meta": {
            "_job_title": rewritten_title if rewritten_title and not rewritten_title.startswith("Error:") else sanitize_text(job_data.get("Job Title", f"Job Listing {index + 1}")),
            "_job_location": location_value,
            "_job_type": job_type_slug,
            "_job_description": rewritten_description if rewritten_description and not rewritten_description.startswith("Error:") else sanitize_text(job_data.get("Job Description", "")),
            "_application": sanitize_text(application, is_url=is_url, is_email=is_email),
            "_job_salary": sanitize_text(job_data.get("job_salary", "")),
            "_job_salary_currency": sanitize_text(job_data.get("job_salary_currency", "")),
            "_job_salary_unit": sanitize_text(job_data.get("job_salary_unit", "")),
            "_company_name": company_name,
            "_company_website": sanitize_text(job_data.get("Company Website", ""), is_url=True),
            "_company_tagline": sanitize_text(job_data.get("Company Details", "")),
            "_company_video": sanitize_text(job_data.get("company_video", ""), is_url=True),
            "_company_twitter": sanitize_text(job_data.get("company_twitter", ""), is_url=True),
            "_company_logo": str(attachment_id) if attachment_id else "",
            "_job_id": job_id
        }
    }
    if region_term_id:
        post_data["job_listing_region"] = [region_term_id]
    if job_type_term_id:
        post_data["job_listing_type"] = [job_type_term_id]
    logger.debug(f"Sending job payload to WordPress for job {index + 1}: {json.dumps(post_data, indent=2)}")
    max_retries = 3
    for attempt in range(max_retries):
        response = None
        try:
            response = session.post(WP_URL, json=post_data, headers=headers, timeout=15, verify=False)
            response.raise_for_status()
            post = response.json()
            logger.info(f"Successfully posted job {index + 1} to WordPress: Post ID {post.get('id')}, URL {post.get('link')}")
            print(f"\nStep 4: Published Job to WordPress")
            print("-" * 30)
            print(f"Job Post ID: {post.get('id')}")
            print(f"Job Post URL: {post.get('link')}")
            save_processed_job_id(job_id, job_url, company_name, job_data.get('URL Page', ''), job_data.get('Job Number', ''))
            return post.get("id"), post.get("link")
        except RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed for job {index + 1}: {e}, Status: {response.status_code if response else 'None'}, Response: {response.text if response else 'None'}")
            print(f"\nStep 4: Attempt {attempt + 1} failed for job {index + 1}: {e}")
            print(f"Response: {response.text if response else 'No response'}")
            print(f"Status Code: {response.status_code if response else 'No status code'}")
            print(f"Response Headers: {response.headers if response else 'No headers'}")
            try:
                check_response = session.get(check_url, headers=headers, timeout=10, verify=False)
                if check_response.status_code == 200:
                    posts = check_response.json()
                    if posts:
                        post = posts[0]
                        logger.info(f"Job {index + 1} (Job ID: {job_id}) was created despite error: Post ID {post.get('id')}, URL {post.get('link')}")
                        print(f"Job {index + 1} was created despite error: Post ID {post.get('id')}, URL {post.get('link')}")
                        save_processed_job_id(job_id, job_url, company_name, job_data.get('URL Page', ''), job_data.get('Job Number', ''))
                        return post.get("id"), post.get("link")
            except RequestException as check_e:
                logger.error(f"Error checking for existing job after failed POST attempt {attempt + 1}: {check_e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying job {index + 1} after {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
    logger.error(f"Failed to post job {index + 1} after {max_retries} attempts.")
    print(f"Failed to post job {index + 1} after {max_retries} attempts.")
    return None, None

def add_three_months_to_date(date_str):
    try:
        date_str = re.sub(r'^Posted:\s*', '', date_str.strip())
        date_obj = datetime.strptime(date_str, '%b %d, %Y')
        new_date = date_obj + timedelta(days=90)
        return new_date.strftime('%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {date_str}, Error: {str(e)}")
        print(f"Invalid date format: {date_str}")
        return None

def scrape_jobs():
    result = []
    processed_job_ids, processed_job_urls, processed_companies = load_mauritius_processed_job_ids()
    start_page = load_last_processed_page()
    for page in range(start_page, start_page + 5):
        url = f'https://www.myjob.mu/ShowResults.aspx?Keywords=&Location=&Category=&Page={page}'
        print(url)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10).text
            soup = BeautifulSoup(resp, 'html.parser')
            job_list = soup.select("#page > div.container > div > div.two-thirds > div > div > div.job-result-logo-title > div > h2 > a")
            urls = ['https://www.myjob.mu' + a.get('href') for a in job_list]
            print(urls)
            for job_url in urls:
                if job_url in processed_job_urls:
                    logger.info(f"Skipping already processed job URL: {job_url}")
                    continue
                data = scrape_job_details(job_url, page)
                if data:
                    result.extend(data)
            save_last_processed_page(page)
        except RequestException as e:
            logger.error(f"Error fetching page {page}: {str(e)}")
            print(f"Error fetching page {page}: {str(e)}")
            continue
    return result

def scrape_job_details(job_url, page):
    res = []
    print(job_url)
    try:
        resp = requests.get(job_url, headers=HEADERS, timeout=10).text
        soup = BeautifulSoup(resp, 'html.parser')

        job_title = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.job-description > h1")
        job_title = job_title.text if job_title else ""
        location = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.job-description > ul > li.location")
        location = location.text if location else "Remote"
        salary = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.job-description > ul > li.salary")
        salary = salary.text if salary else ""
        job_type = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.job-description > ul > li.employment-type")
        job_type = job_type.text if job_type else "Full-time"
        deadline = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.job-description > ul > li.closed-time")
        deadline = deadline.text.replace('Closing', '') if deadline else ""
        company_name = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.company-details > div > h2")
        company_name = company_name.text if company_name else "Unknown Company"
        logo_elements = soup.select("#page > div.container > div > div.three-quarters > div > div.module-content > div.company-details > img")
        logo = [img.get('src') for img in logo_elements]
        location2 = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.company-details > div > ul > li.address")
        location2 = location2.text if location2 else ""
        description = soup.select_one("#page > div.container > div > div.three-quarters > div > div.module-content > div.job-description > div.job-details")
        description = description.text if description else ""
        company_url_elements = soup.select("#page > div.container > div > div.three-quarters > div > div.module-content > div.company-details > div > p > strong > a")
        company_url = ['https://www.myjob.mu' + a.get('href') for a in company_url_elements]
        print(company_url)

        company_name1 = ""
        company_logo = []
        application = ""
        company_website = ""
        company_details = ""
        company_phone = ""

        if company_url:
            try:
                resp = requests.get(company_url[0], headers=HEADERS, timeout=10).text
                soup = BeautifulSoup(resp, 'html.parser')
                company_name1 = soup.select_one("#page > div.container > div > div.three-quarters > div:nth-child(1) > div > div.job-description > h1")
                company_name1 = company_name1.text if company_name1 else ""
                company_logo_elements = soup.select("#page > div.container > div > div.three-quarters > div:nth-child(1) > div > div.job-description > img")
                company_logo = [img.get('src') for img in company_logo_elements]
                application = soup.select_one("#page > div.container > div > div.three-quarters > div:nth-child(1) > div > div.company-details > div:nth-child(1) > ul > li.email-icon")
                application = application.text if application else ""
                company_website = soup.select_one("#page > div.container > div > div.three-quarters > div:nth-child(1) > div > div.company-details > div:nth-child(1) > ul > li.url > a")
                company_website = company_website.text if company_website else ""
                company_details = soup.select_one("#page > div.container > div > div.three-quarters > div:nth-child(1) > div > div.job-description > div")
                company_details = company_details.text if company_details else ""
                company_phone = soup.select_one("#page > div.container > div > div.three-quarters > div:nth-child(1) > div > div.company-details > div:nth-child(1) > ul > li.telnum")
                company_phone = company_phone.text if company_phone else ""
                print("Company URL Obtained:", company_url)
            except Exception as e:
                print("Company URL Failed:", str(e))
                logger.error(f"Error fetching company details from {company_url[0]}: {str(e)}")

        job_id = hashlib.md5(job_url.encode()).hexdigest()
        logger.info(f"Generated Job ID: {job_id}")

        job_title_clean = sanitize_text(job_title)
        job_number = f"job_{page}_{len(res) + 1}"

        separated_info = ["", "", "", job_type]  # Placeholder for category1, category2, state

        job_data = {
            'Job ID': job_id,
            'Job Title': job_title_clean,
            'Job Description': description,
            'Job Type': job_type,
            'Job Qualifications': "",
            'Job Experiences': "",
            'Location': location,
            'Job Fields': "",
            'Date Posted': "",
            'Deadline': deadline,
            'Application': application,
            'Company': company_name,
            'Company Logo': logo[0] if logo else "",
            'Company Industry': "",
            'Company Founded': "",
            'Company Type': "",
            'Company Website': company_website,
            'Company Address': location2,
            'Company Details': company_details,
            'Job URL': job_url,
            'New Date String': add_three_months_to_date(deadline) if deadline else "",
            'Separated Info Company': separated_info[0],
            'Separated Info Location': separated_info[1],
            'Separated Info State': separated_info[2],
            'Separated Info Job Type': separated_info[3],
            'URL Page': str(page),
            'Job Number': job_number
        }

        company_data = {
            'company_name': company_name,
            'company_logo': company_logo[0] if company_logo else "",
            'company_industry': "",
            'company_founded': "",
            'company_type': "",
            'company_website': company_website,
            'company_address': location2,
            'company_details': company_details
        }

        extracted_title = extract_job_title(job_title_clean)
        print(f"\nParaphrasing Job Title and Description for Job ID: {job_id}")
        print("-" * 30)
        print(f"Extracted Job Title: {extracted_title}")
        combined_paraphrased, rewritten_title, rewritten_description = paraphrase_title_and_description(
            extracted_title,
            description,
            len(res),
            max_attempts=5
        )

        processed_companies = load_mauritius_processed_job_ids()[2]
        if company_name not in processed_companies and company_name != "Unknown Company":
            company_post_id, company_post_url = save_company_to_wordpress(len(res), company_data)
            if company_post_id:
                print(f"Successfully posted company {company_name} to WordPress. Post ID: {company_post_id}, URL: {company_post_url}")
            else:
                print(f"Failed to post company {company_name} to WordPress.")

        post_id, post_url = save_article_to_wordpress(len(res), job_data, rewritten_title, rewritten_description, application)
        if post_id:
            print(f"Successfully posted job {job_number} (Job ID: {job_id}, URL: {job_url}) to WordPress. Post ID: {post_id}, URL: {post_url}")
            save_processed_job_id(job_id, job_url, company_name, page, job_number)
        else:
            print(f"Failed to post job {job_number} (Job ID: {job_id}, URL: {job_url}) to WordPress.")
            save_processed_job_id(job_id, job_url, company_name, page, job_number)

        res.append([job_data, company_data])
        logger.info(f"Appended job data for Job ID {job_id}")

        return res
    except requests.RequestException as e:
        logger.error(f"Error scraping job details from {job_url}: {str(e)}")
        print(f"Error scraping job details from {job_url}: {str(e)}")
        return None



def main():
    max_cycles = 10
    cycle_count = 0
    while cycle_count < max_cycles:
        print(f"\nStarting cycle {cycle_count + 1} of job processing...")
        jobs = scrape_jobs()
        logger.info(f"Total jobs scraped in cycle {cycle_count + 1}: {len(jobs)}")
        
        if not jobs:
            logger.info(f"No jobs found in cycle {cycle_count + 1}. Moving to next cycle.")
            print(f"No jobs found in cycle {cycle_count + 1}. Moving to next cycle.")
            cycle_count += 1
            time.sleep(60)
            continue

        for index, job in enumerate(jobs):
            try:
                job_data, company_data = job
                job_id = job_data.get('Job ID', '')
                job_url = job_data.get('Job URL', '')
                company_name = job_data.get('Company', 'Unknown Company')
                logger.info(f"Processing job {index + 1} (Job ID: {job_id}, URL: {job_url})")

                job_title = extract_job_title(job_data.get('Job Title', ''))
                job_description = job_data.get('Job Description', '')
                application = job_data.get('Application', '')

                print(f"\nParaphrasing Job Title and Description for Job ID: {job_id}")
                combined_paraphrased, rewritten_title, rewritten_description = paraphrase_title_and_description(
                    job_title,
                    job_description,
                    index,
                    max_attempts=5
                )

                processed_companies = load_mauritius_processed_job_ids()[2]
                if company_name not in processed_companies and company_name != "Unknown Company":
                    company_post_id, company_post_url = save_company_to_wordpress(index, company_data)
                    if company_post_id:
                        logger.info(f"Posted company {company_name} to WordPress. Post ID: {company_post_id}, URL: {company_post_url}")
                        print(f"Posted company {company_name} to WordPress. Post ID: {company_post_id}, URL: {company_post_url}")
                    else:
                        logger.warning(f"Failed to post company {company_name} to WordPress.")
                        print(f"Failed to post company {company_name} to WordPress.")

                post_id, post_url = save_article_to_wordpress(
                    index,
                    job_data,
                    rewritten_title,
                    rewritten_description,
                    application
                )

                if post_id:
                    logger.info(f"Posted job {index + 1} (Job ID: {job_id}) to WordPress. Post ID: {post_id}, URL: {post_url}")
                    print(f"Posted job {index + 1} (Job ID: {job_id}) to WordPress. Post ID: {post_id}, URL: {post_url}")
                else:
                    logger.error(f"Failed to post job {index + 1} (Job ID: {job_id}) to WordPress.")
                    print(f"Failed to post job {index + 1} (Job ID: {job_id}) to WordPress.")

                save_processed_job_id(
                    job_id,
                    job_url,
                    company_name,
                    job_data.get('URL Page', ''),
                    job_data.get('Job Number', '')
                )

                if (index + 1) % 5 == 0:
                    logger.info("Pausing for 30 seconds after processing 5 jobs.")
                    print("Pausing for 30 seconds after processing 5 jobs.")
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error processing job {index + 1} (Job ID: {job_id}): {str(e)}")
                print(f"Error processing job {index + 1} (Job ID: {job_id}): {str(e)}")
                continue

        cycle_count += 1
        logger.info(f"Completed cycle {cycle_count}. Waiting 60 seconds before next cycle.")
        print(f"Completed cycle {cycle_count}. Waiting 60 seconds before next cycle.")
        time.sleep(60)

    logger.info("Completed all processing cycles.")
    print("Completed all processing cycles.")

if __name__ == "__main__":
    main()
