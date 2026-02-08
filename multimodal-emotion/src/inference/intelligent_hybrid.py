"""
Intelligent Hybrid Emotion Recognition Engine
Automatically combines neural model and rule-based fallback for optimal results
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import time
from collections import deque
from dataclasses import dataclass

from src.inference.engine import EmotionInferenceEngine, PredictionResult
from src.inference.llm_fallback import LLMEmotionDetector

logger = logging.getLogger(__name__)


@dataclass
class HybridPredictionResult(PredictionResult):
    """Extended prediction result with hybrid metadata"""
    model_used: str = "neural"  # "neural", "llm_fallback", or "hybrid"
    model_confidence: float = 0.0
    fallback_confidence: float = 0.0
    reason: str = ""  # Why this model was chosen
    

class IntelligentHybridEngine:
    """
    Smart hybrid system that automatically chooses the best prediction
    
    Strategy:
    1. Run both neural model and LLM fallback
    2. Detect neural model bias (always predicting sadness)
    3. Compare confidence scores
    4. Choose the most reliable prediction
    5. Transparent about which system was used
    """
    
    EMOTIONS = {
        0: "Happiness",
        1: "Sadness",
        2: "Anger"
    }
    
    # Thresholds for intelligent decision-making
    CONFIDENCE_THRESHOLD = 0.60  # Minimum acceptable confidence
    BIAS_THRESHOLD = 0.60  # If >60% predictions are same emotion = biased (lowered from 70%)
    MIN_PREDICTIONS_FOR_BIAS = 3  # Only need 3 predictions to detect bias (faster)
    AGREEMENT_BONUS = 0.10  # Confidence boost if both systems agree
    
    def __init__(
        self,
        model_path: str = "experiments/emotion_pretrained_sota/checkpoint_best.pt",
        use_neural: bool = True
    ):
        """
        Initialize intelligent hybrid engine
        
        Args:
            model_path: Path to neural model checkpoint
            use_neural: Whether to attempt using neural model
        """
        self.use_neural = use_neural
        
        # Initialize LLM fallback (always available)
        logger.info("Initializing LLM fallback system...")
        self.llm_detector = LLMEmotionDetector()
        logger.info("✓ LLM fallback ready")
        
        # Try to initialize neural model
        self.neural_engine = None
        if use_neural:
            try:
                logger.info("Initializing neural model...")
                self.neural_engine = EmotionInferenceEngine(
                    model_path=model_path,
                    enable_attention_viz=True
                )
                logger.info("✓ Neural model ready")
            except Exception as e:
                logger.warning(f"Neural model failed to load: {e}")
                logger.info("Continuing with LLM fallback only")
        
        # Bias detection: track recent predictions
        self.recent_predictions = deque(maxlen=20)  # Last 20 predictions
        self.bias_detected = False
        
        logger.info("✓ Intelligent Hybrid Engine ready")
    
    def _detect_bias(self, emotion: str) -> bool:
        """
        Detect if neural model is biased (always predicting same emotion)
        
        Args:
            emotion: Current prediction
            
        Returns:
            True if bias detected
        """
        self.recent_predictions.append(emotion)
        
        # Need at least MIN_PREDICTIONS_FOR_BIAS predictions to detect bias
        if len(self.recent_predictions) < self.MIN_PREDICTIONS_FOR_BIAS:
            return False
        
        # Count frequency of each emotion
        emotion_counts = {}
        for pred in self.recent_predictions:
            emotion_counts[pred] = emotion_counts.get(pred, 0) + 1
        
        # Check if any emotion dominates (>BIAS_THRESHOLD% of predictions)
        total = len(self.recent_predictions)
        for emotion_name, count in emotion_counts.items():
            frequency = count / total
            if frequency > self.BIAS_THRESHOLD:
                if not self.bias_detected:
                    logger.warning(
                        f"⚠️  BIAS DETECTED: {emotion_name} appears in {frequency:.1%} "
                        f"of last {total} predictions. Switching to LLM fallback."
                    )
                    self.bias_detected = True
                return True
        
        # Bias resolved
        if self.bias_detected:
            logger.info("✓ Bias resolved. Neural model predictions are balanced.")
            self.bias_detected = False
        
        return False
    
    def _choose_best_prediction(
        self,
        neural_result: Optional[PredictionResult],
        llm_result: Any
    ) -> HybridPredictionResult:
        """
        Intelligently choose between neural model and LLM fallback
        
        Decision logic:
        1. If neural model unavailable → use LLM
        2. If bias detected → prefer LLM
        3. If both agree and confident → use neural (higher)
        4. If neural low confidence → use LLM
        5. If LLM higher confidence → use LLM
        6. Otherwise → use neural
        
        Args:
            neural_result: Result from neural model (can be None)
            llm_result: Result from LLM fallback
            
        Returns:
            Best prediction with metadata
        """
        # Case 1: Neural model unavailable
        if neural_result is None:
            return HybridPredictionResult(
                emotion=llm_result.emotion,
                confidence=llm_result.confidence,
                all_probabilities=llm_result.all_probabilities,
                processing_time=llm_result.processing_time,
                attention_weights=llm_result.attention_weights,
                model_used="llm_fallback",
                model_confidence=0.0,
                fallback_confidence=llm_result.confidence,
                reason="Neural model unavailable"
            )
        
        # Check for agreement
        both_agree = neural_result.emotion == llm_result.emotion
        
        # Case 2: Bias detected → strongly prefer LLM (ALWAYS use LLM when biased)
        if self.bias_detected:
            logger.info(f"Using LLM fallback due to detected bias (Neural: {neural_result.emotion}, LLM: {llm_result.emotion})")
            confidence = llm_result.confidence
            if both_agree:
                confidence = min(1.0, confidence + self.AGREEMENT_BONUS)
            
            return HybridPredictionResult(
                emotion=llm_result.emotion,
                confidence=confidence,
                all_probabilities=llm_result.all_probabilities,
                processing_time=llm_result.processing_time,
                attention_weights=llm_result.attention_weights,
                model_used="llm_fallback",
                model_confidence=neural_result.confidence,
                fallback_confidence=llm_result.confidence,
                reason=f"Bias detected: Neural stuck on one emotion. Using LLM."
            )
        
        # Case 3: Both agree and both confident → use neural (trust agreement)
        if both_agree and neural_result.confidence > self.CONFIDENCE_THRESHOLD:
            confidence = min(1.0, neural_result.confidence + self.AGREEMENT_BONUS)
            
            return HybridPredictionResult(
                emotion=neural_result.emotion,
                confidence=confidence,
                all_probabilities=neural_result.all_probabilities,
                processing_time=neural_result.processing_time,
                attention_weights=neural_result.attention_weights,
                model_used="neural",
                model_confidence=neural_result.confidence,
                fallback_confidence=llm_result.confidence,
                reason=f"Both models agree (confidence boost: +{self.AGREEMENT_BONUS:.0%})"
            )
        
        # Case 4: Neural low confidence → use LLM
        if neural_result.confidence < self.CONFIDENCE_THRESHOLD:
            return HybridPredictionResult(
                emotion=llm_result.emotion,
                confidence=llm_result.confidence,
                all_probabilities=llm_result.all_probabilities,
                processing_time=llm_result.processing_time,
                attention_weights=llm_result.attention_weights,
                model_used="llm_fallback",
                model_confidence=neural_result.confidence,
                fallback_confidence=llm_result.confidence,
                reason=f"Neural confidence too low ({neural_result.confidence:.1%} < {self.CONFIDENCE_THRESHOLD:.0%})"
            )
        
        # Case 5: LLM significantly more confident → use LLM
        confidence_diff = llm_result.confidence - neural_result.confidence
        if confidence_diff > 0.15:  # 15% difference
            return HybridPredictionResult(
                emotion=llm_result.emotion,
                confidence=llm_result.confidence,
                all_probabilities=llm_result.all_probabilities,
                processing_time=llm_result.processing_time,
                attention_weights=llm_result.attention_weights,
                model_used="llm_fallback",
                model_confidence=neural_result.confidence,
                fallback_confidence=llm_result.confidence,
                reason=f"LLM more confident (+{confidence_diff:.1%})"
            )
        
        # Case 6: Default to neural model
        return HybridPredictionResult(
            emotion=neural_result.emotion,
            confidence=neural_result.confidence,
            all_probabilities=neural_result.all_probabilities,
            processing_time=neural_result.processing_time,
            attention_weights=neural_result.attention_weights,
            model_used="neural",
            model_confidence=neural_result.confidence,
            fallback_confidence=llm_result.confidence,
            reason="Neural model confident and reliable"
        )
    
    def predict(
        self,
        text: Optional[str] = None,
        audio: Optional[np.ndarray] = None
    ) -> HybridPredictionResult:
        """
        Intelligent prediction using both systems
        
        Args:
            text: Text input
            audio: Audio array (16kHz, mono)
            
        Returns:
            Best prediction with metadata about which system was used
        """
        start_time = time.time()
        
        # Get prediction from LLM fallback (always run this)
        llm_result = self.llm_detector.detect_emotion(text=text or "", audio=audio)
        
        # Get prediction from neural model (if available)
        neural_result = None
        if self.neural_engine:
            try:
                neural_result = self.neural_engine.predict(text=text, audio=audio)
                
                # Detect bias in neural model
                self._detect_bias(neural_result.emotion)
                
            except Exception as e:
                logger.error(f"Neural model prediction failed: {e}")
                logger.info("Falling back to LLM system")
        
        # Intelligently choose best prediction
        final_result = self._choose_best_prediction(neural_result, llm_result)
        
        # Log decision
        logger.info(
            f"Prediction: {final_result.emotion} ({final_result.confidence:.1%}) "
            f"using {final_result.model_used} | Reason: {final_result.reason}"
        )
        
        return final_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        return {
            "neural_available": self.neural_engine is not None,
            "bias_detected": self.bias_detected,
            "recent_predictions": list(self.recent_predictions),
            "prediction_count": len(self.recent_predictions),
            "preferred_system": "llm_fallback" if self.bias_detected else "neural"
        }
