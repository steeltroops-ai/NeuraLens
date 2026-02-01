"""
Pretrained Model Weight Manager v5.1

Handles downloading, caching, and loading of pretrained model weights for:
1. Vessel Segmentation (U-Net / MONAI UNet)
2. DR Classification (EfficientNet-B5)
3. Lesion Detection (optional)

Supports multiple weight sources:
- Local .pth files
- HuggingFace Hub
- Kaggle
- Direct URLs

Author: NeuraLens Medical AI Team
Version: 5.1.0
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Path to the pretrained MONAI segmentation model
PRETRAINED_VESSEL_MODEL_PATH = Path(__file__).parent.parent / "analysis" / "best_metric_model_segmentation2d_dict.pth"

# Check for MONAI availability
try:
    from monai.networks.nets import UNet as MonaiUNet
    MONAI_AVAILABLE = True
    logger.info("MONAI available - using MONAI UNet for vessel segmentation")
except ImportError:
    MONAI_AVAILABLE = False
    logger.warning("MONAI not available - using fallback UNet")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelType(Enum):
    """Supported model types."""
    VESSEL_SEGMENTATION = "vessel_segmentation"
    DR_CLASSIFICATION = "dr_classification"
    LESION_DETECTION = "lesion_detection"
    OPTIC_DISC_SEGMENTATION = "optic_disc_segmentation"


@dataclass
class WeightSource:
    """Configuration for a weight source."""
    model_type: ModelType
    source_type: str  # "huggingface", "url", "local", "timm"
    source_path: str
    filename: str
    expected_hash: Optional[str] = None
    input_size: Tuple[int, int] = (512, 512)
    num_classes: int = 5
    architecture: str = ""
    
    
# Default weight sources (publicly available models)
DEFAULT_WEIGHT_SOURCES = {
    ModelType.VESSEL_SEGMENTATION: WeightSource(
        model_type=ModelType.VESSEL_SEGMENTATION,
        source_type="timm",  # Use timm for easy pretrained loading
        source_path="resnet34",  # Encoder backbone
        filename="vessel_unet_resnet34.pth",
        input_size=(512, 512),
        num_classes=1,
        architecture="unet_resnet34"
    ),
    ModelType.DR_CLASSIFICATION: WeightSource(
        model_type=ModelType.DR_CLASSIFICATION,
        source_type="timm",
        source_path="efficientnet_b5",
        filename="dr_efficientnet_b5.pth",
        input_size=(512, 512),
        num_classes=5,
        architecture="efficientnet_b5"
    ),
}


# Cache directory for downloaded weights
CACHE_DIR = Path(__file__).parent / "weights_cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# U-NET ARCHITECTURE FOR VESSEL SEGMENTATION
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetVesselSegmenter(nn.Module):
    """
    U-Net for vessel segmentation.
    
    Architecture:
    - Encoder: ResNet34-style or simple convolutional
    - Decoder: Upsampling + skip connections
    - Output: Single channel probability map
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list = None):
        super().__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (downsampling)
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=True
                )
            
            x = torch.cat((skip, x), dim=1)  # Skip connection
            x = self.ups[idx + 1](x)  # Conv block
        
        return torch.sigmoid(self.final_conv(x))


# =============================================================================
# DR CLASSIFICATION MODEL
# =============================================================================

class DRClassifier(nn.Module):
    """
    EfficientNet-B5 based DR classifier.
    
    Classifies fundus images into 5 DR grades (ICDR scale):
    - 0: No DR
    - 1: Mild NPDR
    - 2: Moderate NPDR  
    - 3: Severe NPDR
    - 4: Proliferative DR
    """
    
    def __init__(
        self, 
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load backbone using timm
        try:
            import timm
            self.backbone = timm.create_model(
                'efficientnet_b5',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'
            )
            self.feature_dim = self.backbone.num_features
            logger.info(f"Loaded EfficientNet-B5 backbone (features: {self.feature_dim})")
        except ImportError:
            logger.warning("timm not available, using simple CNN backbone")
            self.backbone = self._create_simple_backbone()
            self.feature_dim = 512
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Multi-task heads (optional)
        self.dme_head = nn.Linear(self.feature_dim, 1)  # DME binary
        self.quality_head = nn.Linear(self.feature_dim, 1)  # Quality regression
    
    def _create_simple_backbone(self):
        """Fallback simple CNN if timm not available."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x, return_features: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            return_features: If True, also return feature vector
            
        Returns:
            logits: DR grade logits (B, 5)
            features: Feature vector (B, feature_dim) if return_features=True
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def predict_all(self, x):
        """
        Predict DR grade, DME, and quality.
        
        Returns:
            dict with dr_logits, dme_prob, quality_score
        """
        features = self.backbone(x)
        
        dr_logits = self.classifier(features)
        dme_prob = torch.sigmoid(self.dme_head(features))
        quality_score = torch.sigmoid(self.quality_head(features))
        
        return {
            "dr_logits": dr_logits,
            "dr_probs": torch.softmax(dr_logits, dim=1),
            "dme_prob": dme_prob,
            "quality_score": quality_score,
            "features": features
        }


# =============================================================================
# WEIGHT MANAGER
# =============================================================================

class PretrainedWeightManager:
    """
    Manages downloading and loading of pretrained weights.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self._loaded_models: Dict[ModelType, nn.Module] = {}
    
    def get_vessel_segmenter(
        self, 
        device: torch.device = None,
        force_reload: bool = False
    ) -> nn.Module:
        """
        Get vessel segmentation model.
        
        Priority:
        1. MONAI UNet with pretrained weights from analysis folder
        2. Fallback UNet with random initialization
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if ModelType.VESSEL_SEGMENTATION in self._loaded_models and not force_reload:
            return self._loaded_models[ModelType.VESSEL_SEGMENTATION]
        
        model = None
        
        # Try MONAI model first with pretrained weights
        if MONAI_AVAILABLE and PRETRAINED_VESSEL_MODEL_PATH.exists():
            try:
                logger.info(f"Loading MONAI UNet from {PRETRAINED_VESSEL_MODEL_PATH}")
                
                # Create MONAI UNet with matching architecture
                # Based on the state dict: 5 layers with channels [16, 32, 64, 128, 256]
                model = MonaiUNet(
                    spatial_dims=2,
                    in_channels=3,
                    out_channels=1,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    act='PRELU',
                )
                
                # Load pretrained weights
                state_dict = torch.load(PRETRAINED_VESSEL_MODEL_PATH, map_location=device)
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded pretrained MONAI vessel segmentation model")
                
            except Exception as e:
                logger.warning(f"Failed to load MONAI model: {e}")
                model = None
        
        # Fallback to custom UNet
        if model is None:
            logger.info("Using fallback UNet vessel segmentation model")
            model = UNetVesselSegmenter(in_channels=3, out_channels=1)
            
            # Try to load from cache
            weight_path = self.cache_dir / "vessel_unet.pth"
            if weight_path.exists():
                try:
                    state_dict = torch.load(weight_path, map_location=device)
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded vessel segmentation weights from {weight_path}")
                except Exception as e:
                    logger.warning(f"Failed to load vessel weights: {e}")
        
        model = model.to(device)
        model.eval()
        self._loaded_models[ModelType.VESSEL_SEGMENTATION] = model
        
        return model
    
    def get_dr_classifier(
        self,
        device: torch.device = None,
        force_reload: bool = False
    ) -> DRClassifier:
        """
        Get DR classification model.
        
        Uses EfficientNet-B5 pretrained on ImageNet, optionally fine-tuned for DR.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if ModelType.DR_CLASSIFICATION in self._loaded_models and not force_reload:
            return self._loaded_models[ModelType.DR_CLASSIFICATION]
        
        # Create model with ImageNet pretrained backbone
        model = DRClassifier(num_classes=5, pretrained=True)
        
        # Try to load fine-tuned weights
        weight_path = self.cache_dir / "dr_classifier.pth"
        if weight_path.exists():
            try:
                state_dict = torch.load(weight_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded DR classifier weights from {weight_path}")
            except Exception as e:
                logger.warning(f"Failed to load DR weights: {e}")
        else:
            logger.info("Using ImageNet-pretrained EfficientNet-B5 for DR classification")
        
        model = model.to(device)
        model.eval()
        self._loaded_models[ModelType.DR_CLASSIFICATION] = model
        
        return model
    
    def download_weights(self, model_type: ModelType) -> bool:
        """
        Download pretrained weights for a model.
        
        Currently supports:
        - HuggingFace Hub
        - Direct URL download
        """
        source = DEFAULT_WEIGHT_SOURCES.get(model_type)
        if source is None:
            logger.error(f"No weight source configured for {model_type}")
            return False
        
        if source.source_type == "huggingface":
            return self._download_from_huggingface(source)
        elif source.source_type == "url":
            return self._download_from_url(source)
        elif source.source_type == "timm":
            # Weights are loaded automatically by timm
            logger.info(f"Model {model_type.value} uses timm pretrained weights")
            return True
        
        return False
    
    def _download_from_huggingface(self, source: WeightSource) -> bool:
        """Download weights from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            
            local_path = hf_hub_download(
                repo_id=source.source_path,
                filename=source.filename,
                cache_dir=self.cache_dir
            )
            
            # Copy to our cache
            import shutil
            dest = self.cache_dir / source.filename
            shutil.copy(local_path, dest)
            
            logger.info(f"Downloaded {source.filename} from HuggingFace")
            return True
            
        except ImportError:
            logger.error("huggingface_hub not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            return False
    
    def _download_from_url(self, source: WeightSource) -> bool:
        """Download weights from direct URL."""
        try:
            import urllib.request
            
            dest = self.cache_dir / source.filename
            logger.info(f"Downloading {source.filename}...")
            
            urllib.request.urlretrieve(source.source_path, dest)
            
            # Verify hash if provided
            if source.expected_hash:
                file_hash = self._compute_hash(dest)
                if file_hash != source.expected_hash:
                    logger.error("Hash mismatch - file may be corrupted")
                    dest.unlink()
                    return False
            
            logger.info(f"Downloaded {source.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            return False
    
    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def list_available_models(self) -> Dict[str, bool]:
        """List available models and their status."""
        status = {}
        for model_type in ModelType:
            weight_file = self.cache_dir / f"{model_type.value}.pth"
            status[model_type.value] = weight_file.exists()
        return status


# Singleton instance
weight_manager = PretrainedWeightManager()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_vessel_segmenter(device: torch.device = None) -> UNetVesselSegmenter:
    """Get the vessel segmentation model."""
    return weight_manager.get_vessel_segmenter(device)


def get_dr_classifier(device: torch.device = None) -> DRClassifier:
    """Get the DR classification model."""
    return weight_manager.get_dr_classifier(device)


def download_all_weights():
    """Download all required model weights."""
    for model_type in [ModelType.VESSEL_SEGMENTATION, ModelType.DR_CLASSIFICATION]:
        weight_manager.download_weights(model_type)
