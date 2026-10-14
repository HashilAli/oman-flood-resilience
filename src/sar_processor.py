"""
Sentinel-1 SAR Radar Pre-processing and Water Inundation Mapping
Implements speckle filtering (Lee filter), Otsu / dB thresholding,
and SAR change detection for surface water delineation.
"""
import numpy as np
from scipy.ndimage import uniform_filter, binary_opening, binary_closing
from typing import Dict, Any, Tuple


class Sentinel1SARProcessor:
    """
    Processes Synthetic Aperture Radar (SAR) Sentinel-1 C-band imagery (VV / VH)
    for specular surface water mapping and flood detection.
    """

    def __init__(self, pixel_size_m: float = 30.0):
        self.pixel_size_m = pixel_size_m
        self.pixel_area_km2 = (pixel_size_m * pixel_size_m) / 1e6

    def apply_lee_filter(self, img: np.ndarray, size: int = 5) -> np.ndarray:
        """
        Applies an Enhanced Lee Speckle Filter on SAR intensity/backscatter.
        Reduces radar speckle while preserving sharp water-land boundaries.
        """
        img_mean = uniform_filter(img, size=size)
        img_sqr_mean = uniform_filter(img**2, size=size)
        img_variance = np.maximum(img_sqr_mean - img_mean**2, 1e-6)

        overall_variance = np.var(img)
        weights = img_variance / (img_variance + overall_variance + 1e-6)
        weights = np.clip(weights, 0.0, 1.0)

        filtered = img_mean + weights * (img - img_mean)
        return filtered

    def otsu_threshold(self, image: np.ndarray) -> float:
        """
        Calculates optimal threshold for bimodal SAR backscatter histogram
        (smooth water surface vs rugged rough land).
        """
        flat = image.flatten()
        flat = flat[~np.isnan(flat)]
        
        # Clip extreme outliers for robust binning
        p1, p99 = np.percentile(flat, [1, 99])
        clipped = np.clip(flat, p1, p99)
        
        hist, bin_edges = np.histogram(clipped, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        total = hist.sum()
        current_max, threshold = 0.0, -16.0
        sum_total = np.dot(hist, bin_centers)
        sum_b, weight_b = 0.0, 0.0
        
        for i in range(256):
            weight_b += hist[i]
            if weight_b == 0:
                continue
            weight_f = total - weight_b
            if weight_f == 0:
                break
            
            sum_b += hist[i] * bin_centers[i]
            m_b = sum_b / weight_b
            m_f = (sum_total - sum_b) / weight_f
            
            between_variance = weight_b * weight_f * ((m_b - m_f) ** 2)
            if between_variance > current_max:
                current_max = between_variance
                threshold = bin_centers[i]
                
        # Constrain threshold to realistic SAR calm water range (-22 dB to -14 dB)
        threshold = float(np.clip(threshold, -22.0, -14.0))
        return threshold

    def extract_water_mask(
        self,
        sar_db: np.ndarray,
        threshold_db: float = None,
        filter_speckle: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Extracts binary water mask from SAR backscatter grid.
        Returns:
            (water_mask_boolean_2d, threshold_used_db)
        """
        processed_sar = np.copy(sar_db)
        if filter_speckle:
            processed_sar = self.apply_lee_filter(processed_sar, size=5)

        if threshold_db is None:
            threshold_db = self.otsu_threshold(processed_sar)

        # Calm water shows low backscatter due to specular reflection away from radar
        raw_mask = processed_sar < threshold_db

        # Morphological post-processing to clean radar noise
        # Opening removes isolated single noisy radar dark pixels
        # Closing bridges small gaps
        clean_mask = binary_opening(raw_mask, structure=np.ones((2, 2)))
        clean_mask = binary_closing(clean_mask, structure=np.ones((2, 2)))

        return clean_mask, threshold_db

    def detect_flood_changes(
        self,
        sar_pre: np.ndarray,
        sar_post: np.ndarray,
        fixed_threshold: float = -16.5
    ) -> Dict[str, Any]:
        """
        Performs multi-temporal SAR change detection:
        - Baseline Permanent Water (pre-event)
        - Event-induced Inundation (post-event newly submerged terrain)
        """
        mask_pre, thresh_pre = self.extract_water_mask(sar_pre, threshold_db=fixed_threshold)
        mask_post, thresh_post = self.extract_water_mask(sar_post, threshold_db=fixed_threshold)

        # Newly flooded pixels: water in post, but was dry in pre
        newly_flooded = mask_post & (~mask_pre)
        permanent_water = mask_pre & mask_post

        total_flooded_pixels = int(np.sum(mask_post))
        new_flooded_pixels = int(np.sum(newly_flooded))

        area_total_km2 = total_flooded_pixels * self.pixel_area_km2
        area_new_km2 = new_flooded_pixels * self.pixel_area_km2

        return {
            "mask_pre": mask_pre,
            "mask_post": mask_post,
            "newly_flooded_mask": newly_flooded,
            "permanent_water_mask": permanent_water,
            "threshold_used_db": thresh_post,
            "total_water_area_km2": round(area_total_km2, 3),
            "new_flood_area_km2": round(area_new_km2, 3),
            "total_water_pixels": total_flooded_pixels
        }
