import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pycocoevalcap.cider.cider import Cider

# Download required NLTK packages (uncomment if running first time)
# nltk.download('punkt')
# nltk.download('wordnet')

def calculate_metrics(ground_truth, generated_text):
    """Calculate various text similarity metrics between ground truth and generated text."""
    
    # Tokenize texts
    reference_tokens = word_tokenize(ground_truth.lower())
    candidate_tokens = word_tokenize(generated_text.lower())
    
    # BLEU score with smoothing to handle zero n-gram matches
    reference = [reference_tokens]
    smoothing = SmoothingFunction().method1
    bleu_1 = sentence_bleu(reference, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu(reference, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu(reference, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # METEOR score
    meteor = meteor_score([reference_tokens], candidate_tokens)
    
    # ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(' '.join(candidate_tokens), ' '.join(reference_tokens))[0]
    
    # Cosine similarity with TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ground_truth, generated_text])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Word overlap ratio
    unique_ref = set(reference_tokens)
    unique_cand = set(candidate_tokens)
    overlap = len(unique_ref.intersection(unique_cand))
    overlap_ratio = overlap / len(unique_ref) if len(unique_ref) > 0 else 0
    
    # CIDEr score
    cider_scorer = Cider()
    # Format for CIDEr: {id: [reference]}, {id: [candidate]}
    refs = {'ref1': [ground_truth]}
    hyps = {'ref1': [generated_text]}
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    
    # Return all metrics
    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-4': bleu_4,
        'METEOR': meteor,
        'ROUGE-1': rouge_scores['rouge-1']['f'],
        'ROUGE-2': rouge_scores['rouge-2']['f'],
        'ROUGE-L': rouge_scores['rouge-l']['f'],
        'Cosine Similarity': cosine_sim,
        'Word Overlap Ratio': overlap_ratio,
        'CIDEr': cider_score
    }

# # Input data
# uid = "474"
# ground_truth = "Chest: Stable cardiomediastinal silhouette. Pulmonary vascularity is within normal limits. Hyperlucent apices. Negative for focal airspace disease or consolidation. Negative for pneumothorax or pleural effusion. Healed remote left 9th rib fracture. Right shoulder: Negative for fracture or dislocation."
# low_res_report = "SI XXXX. Mirveated left base cardiomediastinum are clear. No typical scarring and expanded. Bony abnormalities, or pneumothorax or large pleural effusion is seen. No acute bony abnormalities."
# high_res_report = "0 was normal. Mediastinal contour. No acute is normal size and mediastinal silhouettes. The lungs. Small bilateral bone consolidations in the lung are sympt. No acute bibasT. Heart size and mediastinal"

uid = "402"
ground_truth = "The heart and lungs have XXXX XXXX in the interval. Both lungs are clear and expanded. Heart and mediastinum normal."
low_res_report = "the heart size and mediastinal contours are normal the lungs are clear there is no pleural effusion or pneumothorax the are normal there is a calcified granuloma in the right upper lobe"
high_res_report = "Heart size and mediastinal contours are within normal limits. Lungs are clear. No pleural effusions or pneumothoraces."

# uid = ""
# ground_truth = "There are no acute osseous abnormalities. Questionable old left posterior third and fourth rib fractures. Visualized soft tissues are within normal limits. Normal heart size. Normal hilar vascular markings. Subtle prominence of interstitial markings in the bases, left worse than right. No focal area of consolidation, pleural effusion, or pneumothorax. "
# low_res_report = "no focal consolidation no pneumothorax no large pleural effusions no evidence of pneumothorax no acute bony abnormality heart size and mediastinal contours are within normal limits no acute bony abnormality no acute osseous abnormality no acute soft tissue abnormality "
# high_res_report = "The heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. There is no pleural effusion or pneumothorax. There is no acute bony abnormality."

# uid = "8"
# ground_truth = "The heart, pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia. There is an interim XXXX cervical spinal fusion partly evaluated"
# low_res_report = "viewed with xxxx section through the scapular region of the right above rib "
# high_res_report = "viewed upper chest x xxxx, pointing to rule out fracture. no focus of pulmonary pneumonia. the cardiomediastinal contours are within normal limits. the pulmonary vascularity is within normal limits."

# uid = "2070"
# ground_truth = "There are no acute osseous abnormalities. Questionable old left posterior third and fourth rib fractures. Visualized soft tissues are within normal limits. Normal heart size. Normal hilar vascular markings. Subtle prominence of interstitial markings in the bases, left worse than right. No focal area of consolidation, pleural effusion, or pneumothorax."
# low_res_report = "no focal consolidation no pneumothorax no large pleural effusions no evidence of pneumothorax no acute bony abnormality heart size and mediastinal contours are within normal limits no acute bony abnormality no acute osseous abnormality no acute soft tissue abnormality"
# high_res_report = "The heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. There is no pleural effusion or pneumothorax. There is no acute bony abnormality."


# Calculate metrics
print(f"UID: {uid}")
print("\nGround Truth:")
print(ground_truth)
print("\nLow-resolution Report:")
print(low_res_report)

low_res_metrics = calculate_metrics(ground_truth, low_res_report)
print("\nLow-resolution Metrics:")
for metric, value in low_res_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nHigh-resolution Report:")
print(high_res_report)

high_res_metrics = calculate_metrics(ground_truth, high_res_report)
print("\nHigh-resolution Metrics:")
for metric, value in high_res_metrics.items():
    print(f"{metric}: {value:.4f}")

# Compare the results
print("\nMetrics Comparison (High-res vs Low-res):")
for metric in low_res_metrics.keys():
    diff = high_res_metrics[metric] - low_res_metrics[metric]
    print(f"{metric}: {diff:.4f} ({'better' if diff > 0 else 'worse' if diff < 0 else 'same'})")