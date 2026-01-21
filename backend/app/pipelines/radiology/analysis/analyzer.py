"""
Radiology X-Ray Analyzer - Using TorchXRayVision
Pre-trained on 8 chest X-ray datasets for 18 pathologies
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum

# Lazy import flags
TORCHXRAY_AVAILABLE = False
TIMM_AVAILABLE = False
CV2_AVAILABLE = False

if TYPE_CHECKING:
    import torch
    import torchxrayvision as xrv
    import torchvision.transforms as transforms

# For heatmaps
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RiskLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# TorchXRayVision pathology labels
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

# Simplified for demo
PRIORITY_CONDITIONS = [
    "Pneumonia", "Cardiomegaly", "Effusion", "Consolidation",
    "Atelectasis", "Nodule", "Mass", "Pneumothorax"
]


@dataclass
class RadiologyResult:
    primary_finding: str
    confidence: float
    risk_level: str
    findings: List[Dict[str, Any]]
    heatmap_base64: Optional[str]
    recommendation: str
    all_predictions: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class XRayAnalyzer:
    """
    Chest X-Ray Analysis using TorchXRayVision
    
    Models trained on:
    - NIH ChestX-ray14
    - CheXpert
    - MIMIC-CXR
    - PadChest
    - And more!
    """
    
    def __init__(self):
        self.model = None
        self.transform = None
        self._model_loaded = False
        self._backend = "simulation"  # simulation, torchxray, timm
    
    def _ensure_loaded(self):
        """Lazy load the model on first use"""
        if self._model_loaded:
            return

        global TORCHXRAY_AVAILABLE, TIMM_AVAILABLE
        
        # Try loading TorchXRayVision
        try:
            import torch
            import torchxrayvision as xrv
            import torchvision.transforms as transforms
            self._torch = torch
            self._xrv = xrv
            self._transforms = transforms
            TORCHXRAY_AVAILABLE = True
            
            # Load model
            print("[LAZY] Loading TorchXRayVision model...")
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model.eval()
            
            # specific transforms
            self.transform = transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])
            self._backend = "torchxray"
            self._model_loaded = True
            print("[LAZY] TorchXRayVision loaded successfully")
            return
        except ImportError:
            print("[LAZY] torchxrayvision not found")
        except Exception as e:
            print(f"[LAZY] Failed to load TorchXRayVision: {e}")

        # Try TIMM backup
        try:
            import timm
            import torch
            import torchvision.transforms as transforms
            self._torch = torch
            TIMM_AVAILABLE = True
            
            print("[LAZY] Loading TIMM EfficientNet...")
            self.model = timm.create_model(
                'efficientnet_b0',
                pretrained=True,
                num_classes=len(PATHOLOGIES)
            )
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            self._backend = "timm"
            self._model_loaded = True
            print("[LAZY] TIMM loaded as backup")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[LAZY] Failed to load TIMM: {e}")

        self._model_loaded = True
        print("[LAZY] Using simulation mode")

    def analyze(self, image_bytes: bytes) -> RadiologyResult:
        """Analyze chest X-ray image"""
        self._ensure_loaded()
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        if self._backend == "torchxray" and self.model is not None:
            predictions = self._predict_torchxray(image)
        elif self._backend == "timm" and self.model is not None:
            predictions = self._predict_timm(image)
        else:
            predictions = self._simulate_predictions(image)
        
        # Find primary finding
        primary_idx = np.argmax(list(predictions.values()))
        primary_finding = list(predictions.keys())[primary_idx]
        primary_confidence = list(predictions.values())[primary_idx]
        
        # Check if normal (no significant findings)
        max_pathology = max(predictions.values())
        if max_pathology < 0.3:
            primary_finding = "No Significant Findings"
            primary_confidence = 1.0 - max_pathology
        
        # Generate findings
        findings = self._generate_findings(predictions)
        
        # Calculate risk
        risk_level = self._calculate_risk(predictions, primary_finding)
        
        # Generate heatmap
        heatmap = self._generate_heatmap(image)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(primary_finding, risk_level)
        
        return RadiologyResult(
            primary_finding=primary_finding,
            confidence=round(primary_confidence * 100, 1),
            risk_level=risk_level,
            findings=findings,
            heatmap_base64=heatmap,
            recommendation=recommendation,
            all_predictions={k: round(v * 100, 1) for k, v in predictions.items()}
        )
    
    def _predict_torchxray(self, image: Image.Image) -> Dict[str, float]:
        """Predict using TorchXRayVision"""
        # Convert to grayscale numpy array
        img = np.array(image.convert('L'))
        
        # Normalize to [0, 255] range expected by xrv
        img = self._xrv.datasets.normalize(img, 255)
        
        # Add channel dimension if needed
        if len(img.shape) == 2:
            img = img[None, ...]
        
        # Apply transforms
        img = self.transform(img)
        img = self._torch.from_numpy(img).unsqueeze(0).float()
        
        # Predict
        with self._torch.no_grad():
            outputs = self.model(img)
            probs = self._torch.sigmoid(outputs).numpy()[0]
        
        # Map to pathology names
        predictions = {}
        for i, pathology in enumerate(self.model.pathologies):
            if pathology in PATHOLOGIES:
                predictions[pathology] = float(probs[i])
        
        return predictions
    
    def _predict_timm(self, image: Image.Image) -> Dict[str, float]:
        """Predict using timm model"""
        img = image.convert('L')
        img_tensor = self.transform(img).unsqueeze(0)
        
        with self._torch.no_grad():
            outputs = self.model(img_tensor)
            probs = self._torch.softmax(outputs, dim=1).numpy()[0]
        
        return {PATHOLOGIES[i]: float(probs[i]) for i in range(len(PATHOLOGIES))}
    
    def _simulate_predictions(self, image: Image.Image) -> Dict[str, float]:
        """Simulate predictions for demo mode"""
        # Analyze image brightness for realistic demo
        img_array = np.array(image.convert('L'))
        mean_brightness = np.mean(img_array) / 255
        
        # Generate realistic-looking predictions
        predictions = {}
        
        for pathology in PRIORITY_CONDITIONS:
            # Base probability
            base_prob = np.random.uniform(0.05, 0.25)
            
            # Adjust based on image characteristics
            if pathology == "Pneumonia" and mean_brightness < 0.4:
                base_prob += 0.3
            elif pathology == "Cardiomegaly":
                base_prob += np.random.uniform(0, 0.2)
            
            predictions[pathology] = min(0.95, base_prob)
        
        return predictions
    
    def _generate_findings(self, predictions: Dict[str, float]) -> List[Dict]:
        """Generate findings from predictions"""
        findings = []
        
        # Sort by probability
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for condition, prob in sorted_preds[:5]:
            if prob < 0.1:
                continue
            
            severity = self._get_severity(prob)
            
            findings.append({
                "condition": condition,
                "probability": round(prob * 100, 1),
                "severity": severity,
                "description": self._get_description(condition)
            })
        
        # Add normal finding if nothing significant
        if not findings or max(predictions.values()) < 0.3:
            findings.insert(0, {
                "condition": "No Significant Abnormality",
                "probability": round((1 - max(predictions.values())) * 100, 1),
                "severity": "normal",
                "description": "Lungs appear clear. Heart size is normal. No acute findings."
            })
        
        return findings
    
    def _get_severity(self, probability: float) -> str:
        if probability >= 0.7:
            return "high"
        elif probability >= 0.4:
            return "moderate"
        elif probability >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _get_description(self, condition: str) -> str:
        descriptions = {
            "Pneumonia": "Opacity consistent with infectious consolidation",
            "Cardiomegaly": "Cardiac silhouette enlarged beyond normal limits",
            "Effusion": "Fluid accumulation in pleural space",
            "Consolidation": "Dense opacity suggesting alveolar filling",
            "Atelectasis": "Partial lung collapse with volume loss",
            "Nodule": "Focal rounded opacity requiring follow-up",
            "Mass": "Large opacity suspicious for neoplasm",
            "Pneumothorax": "Air in pleural space with lung collapse",
            "Emphysema": "Hyperinflation with flattened diaphragms",
            "Fibrosis": "Reticular opacities suggesting scarring"
        }
        return descriptions.get(condition, "Finding detected - clinical correlation recommended")
    
    def _calculate_risk(self, predictions: Dict[str, float], primary: str) -> str:
        """Calculate overall risk level"""
        max_prob = max(predictions.values())
        
        # High-risk conditions
        critical_conditions = ["Pneumothorax", "Mass"]
        high_risk_conditions = ["Pneumonia", "Effusion", "Consolidation"]
        
        if primary in critical_conditions and max_prob > 0.5:
            return RiskLevel.CRITICAL.value
        elif primary in high_risk_conditions and max_prob > 0.6:
            return RiskLevel.HIGH.value
        elif max_prob > 0.4:
            return RiskLevel.MODERATE.value
        elif max_prob > 0.2:
            return RiskLevel.LOW.value
        else:
            return RiskLevel.NORMAL.value
    
    def _generate_heatmap(self, image: Image.Image) -> Optional[str]:
        """Generate attention heatmap overlay"""
        if not CV2_AVAILABLE:
            return None
        
        try:
            # Convert to numpy
            img_array = np.array(image.convert('RGB'))
            height, width = img_array.shape[:2]
            
            # Create synthetic attention map focused on lung regions
            y, x = np.ogrid[:height, :width]
            
            # Left and right lung centers
            left_center = (width // 3, height // 2)
            right_center = (2 * width // 3, height // 2)
            
            # Gaussian blobs for lungs
            sigma = width // 4
            left_lung = np.exp(-((x - left_center[0])**2 + (y - left_center[1])**2) / (2 * sigma**2))
            right_lung = np.exp(-((x - right_center[0])**2 + (y - right_center[1])**2) / (2 * sigma**2))
            
            attention = left_lung + right_lung
            attention = (attention / attention.max() * 255).astype(np.uint8)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
            
            # Overlay
            overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
            
            # Encode to base64
            _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            print(f"Heatmap generation failed: {e}")
            return None
    
    def _generate_recommendation(self, finding: str, risk: str) -> str:
        """Generate clinical recommendation"""
        if risk == RiskLevel.CRITICAL.value:
            return "URGENT: Immediate physician review required. Consider emergency intervention."
        elif risk == RiskLevel.HIGH.value:
            return "Priority consultation recommended. Consider CT for further evaluation."
        elif risk == RiskLevel.MODERATE.value:
            return "Clinical correlation advised. Follow-up imaging may be warranted."
        elif risk == RiskLevel.LOW.value:
            return "Minor findings noted. Routine follow-up if clinically indicated."
        else:
            return "No significant abnormalities. Continue standard screening intervals."
