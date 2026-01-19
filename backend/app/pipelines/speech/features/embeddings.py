"""
Embedding Extractor
Self-supervised speech embeddings from pretrained models.

Supports:
- Wav2Vec 2.0
- HuBERT
- WavLM
- Whisper encoder embeddings
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
import torch

logger = logging.getLogger(__name__)

# Check for available models
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (
        Wav2Vec2Processor, Wav2Vec2Model,
        WhisperProcessor, WhisperModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Embedding extraction disabled.")


@dataclass
class SpeechEmbeddings:
    """Container for learned speech representations."""
    wav2vec: Optional[np.ndarray] = None      # 768 or 1024 dim
    whisper: Optional[np.ndarray] = None      # 512 or 768 dim
    hubert: Optional[np.ndarray] = None       # 768 or 1024 dim
    wavlm: Optional[np.ndarray] = None        # 768 or 1024 dim
    
    fused: Optional[np.ndarray] = None        # Concatenated/fused
    
    transcription: Optional[str] = None       # Whisper transcription
    word_timestamps: Optional[List[Dict]] = None  # Word-level timing
    
    def get_embedding_dim(self) -> int:
        """Get dimension of fused embedding."""
        if self.fused is not None:
            return len(self.fused)
        return 0
    
    def to_dict(self) -> Dict:
        return {
            "has_wav2vec": self.wav2vec is not None,
            "has_whisper": self.whisper is not None,
            "has_hubert": self.hubert is not None,
            "has_wavlm": self.wavlm is not None,
            "embedding_dim": self.get_embedding_dim(),
            "transcription": self.transcription,
            "has_timestamps": self.word_timestamps is not None
        }


class EmbeddingExtractor:
    """
    Extract self-supervised speech embeddings.
    
    These learned representations capture acoustic patterns
    that may not be captured by handcrafted features.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "auto",
        load_wav2vec: bool = False,
        load_whisper: bool = True,  # Default on - most useful
        load_hubert: bool = False,
        load_wavlm: bool = False
    ):
        self.sample_rate = sample_rate
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Model flags
        self.load_wav2vec = load_wav2vec
        self.load_whisper = load_whisper
        self.load_hubert = load_hubert
        self.load_wavlm = load_wavlm
        
        # Model instances (lazy loaded)
        self._wav2vec_processor = None
        self._wav2vec_model = None
        self._whisper_processor = None
        self._whisper_model = None
        
        self._models_loaded = False
        
    async def initialize(self):
        """Lazy load models."""
        if self._models_loaded or not TRANSFORMERS_AVAILABLE:
            return
            
        try:
            if self.load_whisper:
                logger.info("Loading Whisper model...")
                self._whisper_processor = WhisperProcessor.from_pretrained(
                    "openai/whisper-tiny"
                )
                self._whisper_model = WhisperModel.from_pretrained(
                    "openai/whisper-tiny"
                ).to(self.device)
                self._whisper_model.eval()
                logger.info("Whisper loaded.")
                
            # Wav2Vec loading (optional - heavier)
            if self.load_wav2vec:
                logger.info("Loading Wav2Vec2 model...")
                self._wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self._wav2vec_model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                ).to(self.device)
                self._wav2vec_model.eval()
                logger.info("Wav2Vec2 loaded.")
                
            self._models_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
    
    async def extract(
        self, 
        audio: np.ndarray,
        extract_transcription: bool = True
    ) -> SpeechEmbeddings:
        """
        Extract speech embeddings from audio.
        
        Args:
            audio: Normalized audio array
            extract_transcription: Whether to also transcribe
            
        Returns:
            SpeechEmbeddings with available representations
        """
        if not TRANSFORMERS_AVAILABLE:
            return SpeechEmbeddings()
            
        await self.initialize()
        
        embeddings = SpeechEmbeddings()
        
        try:
            # Whisper embeddings and transcription
            if self._whisper_model is not None:
                embeddings = await self._extract_whisper(
                    audio, embeddings, extract_transcription
                )
                
            # Wav2Vec embeddings
            if self._wav2vec_model is not None:
                embeddings = await self._extract_wav2vec(audio, embeddings)
                
            # Fuse available embeddings
            embeddings = self._fuse_embeddings(embeddings)
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            
        return embeddings
    
    async def _extract_whisper(
        self,
        audio: np.ndarray,
        embeddings: SpeechEmbeddings,
        extract_transcription: bool
    ) -> SpeechEmbeddings:
        """Extract Whisper encoder embeddings."""
        try:
            with torch.no_grad():
                # Prepare input
                input_features = self._whisper_processor(
                    audio,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Get encoder hidden states
                encoder_outputs = self._whisper_model.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state
                
                # Mean pooling over time
                embeddings.whisper = hidden_states.mean(dim=1).cpu().numpy().flatten()
                
        except Exception as e:
            logger.warning(f"Whisper embedding extraction failed: {e}")
            
        return embeddings
    
    async def _extract_wav2vec(
        self,
        audio: np.ndarray,
        embeddings: SpeechEmbeddings
    ) -> SpeechEmbeddings:
        """Extract Wav2Vec2 embeddings."""
        try:
            with torch.no_grad():
                # Prepare input
                inputs = self._wav2vec_processor(
                    audio,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get hidden states
                outputs = self._wav2vec_model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Mean pooling over time
                embeddings.wav2vec = hidden_states.mean(dim=1).cpu().numpy().flatten()
                
        except Exception as e:
            logger.warning(f"Wav2Vec embedding extraction failed: {e}")
            
        return embeddings
    
    def _fuse_embeddings(self, embeddings: SpeechEmbeddings) -> SpeechEmbeddings:
        """Concatenate available embeddings into fused representation."""
        parts = []
        
        if embeddings.wav2vec is not None:
            parts.append(embeddings.wav2vec)
        if embeddings.whisper is not None:
            parts.append(embeddings.whisper)
        if embeddings.hubert is not None:
            parts.append(embeddings.hubert)
        if embeddings.wavlm is not None:
            parts.append(embeddings.wavlm)
            
        if parts:
            embeddings.fused = np.concatenate(parts)
            
        return embeddings
    
    def unload(self):
        """Unload models to free memory."""
        self._wav2vec_model = None
        self._wav2vec_processor = None
        self._whisper_model = None
        self._whisper_processor = None
        self._models_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
