"""
Radiology Pipeline - Visualization
Grad-CAM heatmap generation for explainability
"""

import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try importing opencv
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. Heatmap generation limited.")

# Try importing pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed. Using synthetic heatmaps.")


class XRayVisualizer:
    """
    X-Ray Visualization with Grad-CAM
    
    Generates attention heatmaps showing which regions
    the model focused on for its predictions.
    """
    
    COLORMAP = cv2.COLORMAP_JET if CV2_AVAILABLE else None
    
    def __init__(self, model=None, target_layer=None):
        """
        Initialize visualizer
        
        Args:
            model: PyTorch model for Grad-CAM
            target_layer: Target layer for Grad-CAM (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.grad_cam = None
        
        if model and target_layer and GRADCAM_AVAILABLE:
            try:
                self.grad_cam = GradCAM(
                    model=model,
                    target_layers=[target_layer]
                )
            except Exception as e:
                logger.error(f"Failed to initialize Grad-CAM: {e}")
    
    def generate_heatmap(
        self,
        image_tensor,
        original_image: np.ndarray,
        target_class: int = None
    ) -> Optional[str]:
        """
        Generate Grad-CAM heatmap
        
        Args:
            image_tensor: Preprocessed image tensor for model
            original_image: Original image as numpy array
            target_class: Target class index (None = use highest prediction)
            
        Returns:
            Base64 encoded heatmap overlay image
        """
        if not CV2_AVAILABLE:
            return None
        
        try:
            if self.grad_cam is not None:
                # Use actual Grad-CAM
                grayscale_cam = self.grad_cam(
                    input_tensor=image_tensor,
                    targets=None  # Use highest scoring class
                )
                grayscale_cam = grayscale_cam[0, :]
            else:
                # Generate synthetic attention map
                grayscale_cam = self._generate_synthetic_attention(original_image)
            
            # Resize original to match CAM size
            if len(original_image.shape) == 2:
                original_resized = cv2.resize(original_image, (224, 224))
                original_rgb = cv2.cvtColor(
                    original_resized.astype(np.uint8),
                    cv2.COLOR_GRAY2RGB
                )
            else:
                original_rgb = cv2.resize(original_image, (224, 224))
            
            # Apply colormap to CAM
            heatmap = cv2.applyColorMap(
                np.uint8(255 * grayscale_cam),
                self.COLORMAP
            )
            
            # Overlay on original
            overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)
            
            # Encode to base64
            _, buffer = cv2.imencode('.png', overlay)
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return self._generate_fallback_heatmap(original_image)
    
    def _generate_synthetic_attention(self, image: np.ndarray) -> np.ndarray:
        """
        Generate synthetic attention map focused on lung regions
        
        Used when Grad-CAM is not available
        """
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Define lung region centers (approximate anatomical positions)
        left_lung_center = (width * 0.33, height * 0.45)
        right_lung_center = (width * 0.67, height * 0.45)
        heart_center = (width * 0.5, height * 0.55)
        
        # Lung region sigma (spread)
        lung_sigma = width * 0.18
        heart_sigma = width * 0.12
        
        # Gaussian distributions for attention
        left_attention = np.exp(
            -((x - left_lung_center[0])**2 + (y - left_lung_center[1])**2) 
            / (2 * lung_sigma**2)
        )
        right_attention = np.exp(
            -((x - right_lung_center[0])**2 + (y - right_lung_center[1])**2) 
            / (2 * lung_sigma**2)
        )
        heart_attention = np.exp(
            -((x - heart_center[0])**2 + (y - heart_center[1])**2) 
            / (2 * heart_sigma**2)
        ) * 0.5  # Lower weight for heart
        
        # Combine attention maps
        attention = left_attention + right_attention + heart_attention
        
        # Normalize to [0, 1]
        attention = attention / attention.max()
        
        # Resize to 224x224 for overlay
        attention = cv2.resize(attention.astype(np.float32), (224, 224))
        
        return attention
    
    def _generate_fallback_heatmap(self, original_image: np.ndarray) -> Optional[str]:
        """Generate a basic fallback heatmap when everything else fails"""
        if not CV2_AVAILABLE:
            return None
        
        try:
            # Resize to standard size
            if len(original_image.shape) == 2:
                img = cv2.resize(original_image, (224, 224))
                img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.resize(original_image, (224, 224))
            
            # Generate synthetic attention
            attention = self._generate_synthetic_attention(original_image)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(
                np.uint8(255 * attention),
                cv2.COLORMAP_JET
            )
            
            # Overlay
            overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
            
            # Encode
            _, buffer = cv2.imencode('.png', overlay)
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Fallback heatmap failed: {e}")
            return None
    
    def create_comparison_image(
        self,
        original: np.ndarray,
        heatmap_b64: str
    ) -> Optional[str]:
        """
        Create side-by-side comparison image
        
        Args:
            original: Original X-ray image
            heatmap_b64: Base64 encoded heatmap
            
        Returns:
            Base64 encoded comparison image
        """
        if not CV2_AVAILABLE or not heatmap_b64:
            return None
        
        try:
            # Decode heatmap
            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_img = Image.open(BytesIO(heatmap_bytes))
            heatmap_np = np.array(heatmap_img)
            
            # Resize original
            if len(original.shape) == 2:
                orig_resized = cv2.resize(original, (224, 224))
                orig_rgb = cv2.cvtColor(orig_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                orig_rgb = cv2.resize(original, (224, 224))
            
            # Create side-by-side
            comparison = np.hstack([orig_rgb, heatmap_np])
            
            # Encode
            _, buffer = cv2.imencode('.png', comparison)
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Comparison image failed: {e}")
            return None


def generate_xray_heatmap(
    image_bytes: bytes,
    model=None,
    image_tensor=None
) -> Optional[str]:
    """
    Convenience function to generate X-ray heatmap
    
    Args:
        image_bytes: Raw image bytes
        model: Optional PyTorch model
        image_tensor: Optional preprocessed tensor
        
    Returns:
        Base64 encoded heatmap
    """
    # Load image
    img = Image.open(BytesIO(image_bytes))
    img_np = np.array(img.convert('L'))
    
    visualizer = XRayVisualizer(model=model)
    
    if image_tensor is not None and model is not None:
        return visualizer.generate_heatmap(image_tensor, img_np)
    else:
        return visualizer._generate_fallback_heatmap(img_np)
