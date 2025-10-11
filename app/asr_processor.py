"""
ASR (Automatic Speech Recognition) processor for analyzing speech samples
"""
import os
import re
import whisper
import torch
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import MODEL_MAP, FREE_MODELS

class SpeechAnalyzer:
    """Analyze speech samples and identify disorders."""
    
    def __init__(self):
        """Initialize Whisper model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("base", device=self.device)
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, any]:
        """
        Transcribe audio file using Whisper.
        
        Returns:
            Dict with transcript, confidence, duration, etc.
        """
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False
            )
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "success": True
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def deidentify_transcript(self, transcript: str) -> str:
        """
        Remove personally identifiable information from transcript.
        """
        deidentified = transcript
        
        # Replace "My name is X" patterns
        deidentified = re.sub(
            r"(my name is|i'm|i am|this is)\s+([A-Z][a-z]+)",
            r"\1 [NAME]",
            deidentified,
            flags=re.IGNORECASE
        )
        
        # Replace standalone capitalized names (heuristic)
        words = deidentified.split()
        deidentified_words = []
        for i, word in enumerate(words):
            if (word[0].isupper() and 
                i > 0 and 
                words[i-1][-1] not in '.!?' and
                len(word) > 2 and
                word.lower() not in ['i', 'the', 'a', 'an']):
                deidentified_words.append("[NAME]")
            else:
                deidentified_words.append(word)
        
        deidentified = " ".join(deidentified_words)
        
        # Replace phone numbers
        deidentified = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            deidentified
        )
        
        # Replace email addresses
        deidentified = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            deidentified
        )
        
        # Replace addresses
        deidentified = re.sub(
            r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
            '[ADDRESS]',
            deidentified,
            flags=re.IGNORECASE
        )
        
        # Replace dates of birth
        deidentified = re.sub(
            r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](\d{2}|\d{4})\b',
            '[DATE]',
            deidentified
        )
        
        return deidentified
    
    def detect_fluency_disfluencies(self, text: str) -> List[Dict[str, any]]:
        """
        Detect fluency disfluencies: sound repetitions, syllable repetitions, 
        prolongations, and blocks.
        
        Returns:
            List of disfluency events with type and example
        """
        disfluencies = []
        
        # Pattern 1: Sound repetitions (b-b-ball, s-s-see)
        # Look for single letter followed by hyphen, repeated
        sound_reps = re.finditer(
            r'\b([a-z])-\1+(?:-\1+)*-(\w+)\b',
            text.lower()
        )
        for match in sound_reps:
            sound = match.group(1)
            word = match.group(2)
            full_match = match.group(0)
            count = full_match.count(f'{sound}-')
            disfluencies.append({
                "type": "Sound Repetition",
                "severity": "moderate" if count <= 2 else "severe",
                "example": full_match,
                "count": count,
                "phoneme": sound
            })
        
        # Pattern 2: Syllable repetitions (ba-ba-ball, mo-mo-mommy)
        # Look for CV or CVC syllable patterns repeated
        syllable_reps = re.finditer(
            r'\b([a-z]{2,3})-\1+(?:-\1+)*-?(\w+)?\b',
            text.lower()
        )
        for match in syllable_reps:
            syllable = match.group(1)
            remaining = match.group(2) if match.group(2) else ""
            full_match = match.group(0)
            count = full_match.count(f'{syllable}-')
            
            # Exclude if it's just the same syllable repeated as whole words
            if not remaining and syllable.isalpha():
                continue
                
            disfluencies.append({
                "type": "Syllable Repetition",
                "severity": "moderate" if count <= 2 else "severe",
                "example": full_match,
                "count": count,
                "syllable": syllable
            })
        
        # Pattern 3: Prolongations (sssee, mmmom)
        # Look for repeated letters (but not double letters in normal words)
        prolongations = re.finditer(
            r'\b(\w*?)([a-z])\2{2,}(\w*)\b',
            text.lower()
        )
        
        # Common words with double letters to exclude
        double_letter_words = {
            'see', 'too', 'look', 'good', 'book', 'school', 'room', 
            'ball', 'all', 'call', 'balloon', 'letter', 'better'
        }
        
        for match in prolongations:
            full_word = match.group(0)
            repeated_letter = match.group(2)
            
            # Skip common double-letter words
            if full_word in double_letter_words:
                continue
            
            # Skip if it's a normal double letter in the middle of a word
            if match.group(1) and match.group(3):
                continue
                
            count = len(match.group(0)) - len(match.group(0).replace(repeated_letter, '', 
                       match.group(0).count(repeated_letter) - 1))
            
            disfluencies.append({
                "type": "Prolongation",
                "severity": "moderate" if count <= 3 else "severe",
                "example": full_word,
                "count": count,
                "phoneme": repeated_letter
            })
        
        # Pattern 4: Interjections and fillers (um, uh, like, you know)
        fillers = ['um', 'uh', 'umm', 'uhh', 'er', 'ah', 'like', 'you know', 'I mean']
        text_lower = text.lower()
        
        for filler in fillers:
            count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
            if count > 0:
                disfluencies.append({
                    "type": "Interjection/Filler",
                    "severity": "mild" if count <= 3 else "moderate",
                    "example": filler,
                    "count": count,
                    "filler": filler
                })
        
        # Pattern 5: Blocks (indicated by ellipses or pauses in transcript)
        blocks = re.finditer(r'\.{2,}|\.\.\s', text)
        block_count = len(list(blocks))
        if block_count > 0:
            disfluencies.append({
                "type": "Block/Pause",
                "severity": "moderate" if block_count <= 2 else "severe",
                "example": "...",
                "count": block_count
            })
        
        return disfluencies
    
    def calculate_fluency_metrics(self, disfluencies: List[Dict], 
                                  total_words: int, 
                                  duration: float) -> Dict[str, any]:
        """
        Calculate fluency metrics from detected disfluencies.
        
        Returns:
            Dict with metrics like stuttering frequency, severity, etc.
        """
        if total_words == 0:
            return {
                "stuttering_frequency": 0,
                "severity_rating": "None",
                "total_disfluencies": 0
            }
        
        # Count different types
        sound_reps = sum(1 for d in disfluencies if d["type"] == "Sound Repetition")
        syllable_reps = sum(1 for d in disfluencies if d["type"] == "Syllable Repetition")
        prolongations = sum(1 for d in disfluencies if d["type"] == "Prolongation")
        blocks = sum(1 for d in disfluencies if d["type"] == "Block/Pause")
        fillers = sum(1 for d in disfluencies if d["type"] == "Interjection/Filler")
        
        # Core stuttering behaviors (exclude fillers for severity)
        core_disfluencies = sound_reps + syllable_reps + prolongations + blocks
        total_disfluencies = len(disfluencies)
        
        # Calculate percentage of stuttered syllables
        # Rough estimate: assume 1.5 syllables per word
        estimated_syllables = total_words * 1.5
        stuttering_frequency = (core_disfluencies / estimated_syllables * 100) if estimated_syllables > 0 else 0
        
        # Severity rating based on stuttering frequency
        # Based on clinical guidelines: <3% mild, 3-8% moderate, >8% severe
        if stuttering_frequency < 3:
            severity_rating = "Mild"
        elif stuttering_frequency < 8:
            severity_rating = "Moderate"
        else:
            severity_rating = "Severe"
        
        # If no core disfluencies, check if only fillers
        if core_disfluencies == 0 and fillers > 0:
            severity_rating = "Mild (fillers only)"
        elif core_disfluencies == 0:
            severity_rating = "None"
        
        # Calculate speech rate (words per minute)
        speech_rate = (total_words / duration * 60) if duration > 0 else 0
        
        return {
            "stuttering_frequency": round(stuttering_frequency, 2),
            "severity_rating": severity_rating,
            "total_disfluencies": total_disfluencies,
            "core_disfluencies": core_disfluencies,
            "sound_repetitions": sound_reps,
            "syllable_repetitions": syllable_reps,
            "prolongations": prolongations,
            "blocks": blocks,
            "fillers": fillers,
            "speech_rate_wpm": round(speech_rate, 1)
        }
    
    def analyze_speech_patterns(self, transcript: str, segments: list) -> Dict[str, any]:
        """
        Analyze speech patterns from transcript and segments.
        
        Returns:
            Dict with identified patterns, errors, characteristics
        """
        patterns = {
            "articulation_errors": [],
            "phonological_patterns": [],
            "fluency_issues": [],
            "fluency_disfluencies": [],
            "fluency_metrics": {},
            "language_patterns": [],
            "characteristics": []
        }
        
        text_lower = transcript.lower()
        
        # Detect articulation issues (common substitutions)
        articulation_patterns = {
            r'\bw[aeiouy]': 'Possible /r/ → /w/ substitution (e.g., "wabbit" for "rabbit")',
            r'\bt[aeiouy]': 'Possible /k/ → /t/ substitution (fronting)',
            r'\bd[aeiouy]': 'Possible /g/ → /d/ substitution (fronting)',
            r'\bf[aeiouy]': 'Possible /θ/ (th) → /f/ substitution',
            r'\bs[aeiouy]': 'Possible /θ/ (th) → /s/ substitution',
        }
        
        for pattern, description in articulation_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                patterns["articulation_errors"].append(
                    f"{description} - Found {len(matches)} instance(s)"
                )
        
        # Detect phonological patterns
        # Final consonant deletion
        if re.search(r'\b\w+[aeiouy]\b(?!\w)', text_lower):
            patterns["phonological_patterns"].append(
                "Possible final consonant deletion (words ending in vowels)"
            )
        
        # Cluster reduction
        cluster_pattern = r'\b\w*?([bdgkpt][lr]|s[ptk]|[bdgptk]w)\w*\b'
        if re.search(cluster_pattern, text_lower):
            patterns["phonological_patterns"].append(
                "Check for cluster reduction in complex onset words"
            )
        
        # Detect fluency issues - IMPROVED
        fluency_disfluencies = self.detect_fluency_disfluencies(transcript)
        patterns["fluency_disfluencies"] = fluency_disfluencies
        
        # Calculate total words and duration
        words = transcript.split()
        total_words = len(words)
        duration = segments[-1]['end'] if segments and len(segments) > 0 else 0
        
        # Calculate fluency metrics
        fluency_metrics = self.calculate_fluency_metrics(
            fluency_disfluencies,
            total_words,
            duration
        )
        patterns["fluency_metrics"] = fluency_metrics
        
        # Create summary of fluency issues
        if fluency_disfluencies:
            summary_items = []
            
            if fluency_metrics["sound_repetitions"] > 0:
                examples = [d["example"] for d in fluency_disfluencies if d["type"] == "Sound Repetition"]
                summary_items.append(
                    f"Sound repetitions: {fluency_metrics['sound_repetitions']} (e.g., {', '.join(examples[:3])})"
                )
            
            if fluency_metrics["syllable_repetitions"] > 0:
                examples = [d["example"] for d in fluency_disfluencies if d["type"] == "Syllable Repetition"]
                summary_items.append(
                    f"Syllable repetitions: {fluency_metrics['syllable_repetitions']} (e.g., {', '.join(examples[:3])})"
                )
            
            if fluency_metrics["prolongations"] > 0:
                examples = [d["example"] for d in fluency_disfluencies if d["type"] == "Prolongation"]
                summary_items.append(
                    f"Prolongations: {fluency_metrics['prolongations']} (e.g., {', '.join(examples[:3])})"
                )
            
            if fluency_metrics["blocks"] > 0:
                summary_items.append(
                    f"Blocks/Pauses: {fluency_metrics['blocks']}"
                )
            
            if fluency_metrics["fillers"] > 0:
                summary_items.append(
                    f"Fillers: {fluency_metrics['fillers']}"
                )
            
            summary_items.append(
                f"Stuttering frequency: {fluency_metrics['stuttering_frequency']}% - {fluency_metrics['severity_rating']}"
            )
            
            patterns["fluency_issues"] = summary_items
        else:
            patterns["fluency_issues"] = ["No significant fluency disfluencies detected"]
        
        # Detect language patterns
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        if avg_word_length < 3.5:
            patterns["language_patterns"].append(
                "Short average word length (may indicate language delay)"
            )
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length < 5:
            patterns["language_patterns"].append(
                f"Short sentence structure (avg {avg_sentence_length:.1f} words/sentence)"
            )
        
        # MLU (Mean Length of Utterance) approximation
        if avg_sentence_length < 3:
            patterns["language_patterns"].append(
                "Low MLU - may indicate expressive language delay"
            )
        
        # Overall characteristics
        patterns["characteristics"] = [
            f"Total words: {total_words}",
            f"Total duration: {duration:.1f}s",
            f"Speech rate: {fluency_metrics.get('speech_rate_wpm', 0):.1f} words/min",
            f"Average word length: {avg_word_length:.1f} characters",
            f"Average sentence length: {avg_sentence_length:.1f} words",
            f"Number of sentences: {len(sentences)}"
        ]
        
        return patterns
    
    def identify_disorders_ai(self, transcript: str, patterns: Dict, 
                             model_name: str = "Llama3.2") -> Dict[str, any]:
        """
        Use AI to analyze speech patterns and identify likely disorders.
        """
        # Build detailed fluency section if present
        fluency_section = ""
        if patterns.get("fluency_disfluencies"):
            metrics = patterns.get("fluency_metrics", {})
            disfluencies = patterns.get("fluency_disfluencies", [])
            
            fluency_section = f"""
**Fluency Analysis:**
- Stuttering Frequency: {metrics.get('stuttering_frequency', 0)}%
- Severity: {metrics.get('severity_rating', 'Unknown')}
- Core Disfluencies: {metrics.get('core_disfluencies', 0)}
  - Sound Repetitions: {metrics.get('sound_repetitions', 0)}
  - Syllable Repetitions: {metrics.get('syllable_repetitions', 0)}
  - Prolongations: {metrics.get('prolongations', 0)}
  - Blocks: {metrics.get('blocks', 0)}
- Fillers/Interjections: {metrics.get('fillers', 0)}
- Speech Rate: {metrics.get('speech_rate_wpm', 0)} words/minute

**Detailed Disfluency Examples:**
{chr(10).join(f'- {d["type"]}: "{d["example"]}" (occurred {d.get("count", 1)} times)' for d in disfluencies[:10])}
"""
        
        # Build analysis prompt
        prompt = f"""You are an expert Speech-Language Pathologist analyzing a speech sample.

**Deidentified Transcript:**
{transcript}

**Detected Patterns:**
- Articulation Errors: {', '.join(patterns['articulation_errors']) if patterns['articulation_errors'] else 'None detected'}
- Phonological Patterns: {', '.join(patterns['phonological_patterns']) if patterns['phonological_patterns'] else 'None detected'}
{fluency_section if fluency_section else '- Fluency Issues: ' + (', '.join(patterns['fluency_issues']) if patterns['fluency_issues'] else 'None detected')}
- Language Patterns: {', '.join(patterns['language_patterns']) if patterns['language_patterns'] else 'None detected'}

**Speech Characteristics:**
{chr(10).join(f'- {c}' for c in patterns['characteristics'])}

Based on this comprehensive analysis, provide:

1. **Primary Disorder(s)**: speech (articulation, phonological, or fluency), language (receptive, expressive, pragmatic), or other
2. **Severity Rating**: Mild, Moderate, or Severe with clinical reasoning
3. **Estimated Age Range**: Based on language complexity and speech patterns
4. **Recommended IEP Goals**: 1-3 specific, measurable goals
5. **Clinical Observations**: Key observations for documentation
6. **Therapy Recommendations**: Suggested intervention approaches

Format your response in a structured, professional manner suitable for clinical documentation.
"""
        
        try:
            # Use appropriate LLM
            if model_name in FREE_MODELS:
                llm = ChatOllama(
                    model=MODEL_MAP.get(model_name, "llama3.2:latest"),
                    temperature=0.3
                )
            else:
                llm = ChatOpenAI(
                    model=MODEL_MAP.get(model_name, "gpt-4o"),
                    temperature=0.3
                )
            
            response = llm.invoke(prompt)
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis": analysis,
                "success": True
            }
        except Exception as e:
            return {
                "analysis": f"Error in AI analysis: {str(e)}",
                "success": False
            }

# Global instance
_speech_analyzer = None

def get_speech_analyzer():
    """Get or create speech analyzer instance."""
    global _speech_analyzer
    if _speech_analyzer is None:
        _speech_analyzer = SpeechAnalyzer()
    return _speech_analyzer