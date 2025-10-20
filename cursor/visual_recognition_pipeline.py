#!/usr/bin/env python3
"""
Visual Recognition Pipeline for Image-to-Video Generation
Analyzes input image to understand objects/features before generation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import os


class VisualRecognitionPipeline:
    """
    Analyzes images to extract visual features and enhance prompts
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Args:
            gemini_api_key: Optional Gemini API key for VLM analysis
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.vlm_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ Gemini VLM initialized")
        else:
            self.vlm_model = None
            print("⚠ No Gemini API key - using vision-only analysis")
    
    def analyze_image(self, image_path: str) -> Dict[str, any]:
        """
        Comprehensive image analysis
        
        Returns:
            Dictionary with:
            - red_dots: List of detected red dot locations
            - spatial_description: Text describing layout
            - vlm_description: Gemini's understanding (if available)
            - enhanced_prompt: Suggested prompt enhancement
        """
        print(f"\n🔍 Analyzing image: {image_path}")
        
        # Load image
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        results = {
            'image_size': (width, height),
            'image_path': image_path
        }
        
        # 1. Computer Vision Analysis
        cv_results = self._cv_analysis(image_rgb)
        results.update(cv_results)
        
        # 2. VLM Analysis (if available)
        if self.vlm_model:
            vlm_results = self._vlm_analysis(image_path, cv_results)
            results.update(vlm_results)
        
        # 3. Generate Enhanced Prompt
        enhanced_prompt = self._generate_enhanced_prompt(results)
        results['enhanced_prompt'] = enhanced_prompt
        
        return results
    
    def _cv_analysis(self, image_rgb: np.ndarray) -> Dict[str, any]:
        """
        Computer vision-based analysis
        """
        print("  📊 Running CV analysis...")
        
        height, width = image_rgb.shape[:2]
        results = {}
        
        # Detect red dots
        red_dots = self._detect_red_dots(image_rgb)
        results['red_dots'] = red_dots
        results['num_red_dots'] = len(red_dots)
        
        print(f"    Found {len(red_dots)} red dots")
        
        # Spatial distribution
        if len(red_dots) > 0:
            spatial_desc = self._describe_spatial_distribution(red_dots, width, height)
            results['spatial_distribution'] = spatial_desc
            print(f"    Distribution: {spatial_desc}")
        
        # Color analysis
        color_info = self._analyze_colors(image_rgb)
        results['color_info'] = color_info
        
        # Detect other visual elements
        edges = cv2.Canny(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 50, 150)
        results['has_lines'] = np.sum(edges > 0) > 1000
        results['has_shapes'] = self._detect_shapes(image_rgb)
        
        return results
    
    def _detect_red_dots(self, image_rgb: np.ndarray) -> List[Dict[str, any]]:
        """
        Detect red dots with detailed information
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Red color ranges in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 5:  # Too small
                continue
            
            # Get properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            dots.append({
                'id': i,
                'center': (cx, cy),
                'radius': int(radius),
                'area': area,
                'position': (int(x), int(y))
            })
        
        # Sort by position (left to right, top to bottom)
        dots.sort(key=lambda d: (d['center'][1] // 50, d['center'][0]))
        
        return dots
    
    def _describe_spatial_distribution(self, dots: List[Dict], width: int, height: int) -> str:
        """
        Describe where dots are located
        """
        if len(dots) == 0:
            return "no dots"
        
        # Analyze distribution
        positions = [d['center'] for d in dots]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # Spatial regions
        left = sum(1 for x in xs if x < width * 0.33)
        center = sum(1 for x in xs if width * 0.33 <= x < width * 0.67)
        right = sum(1 for x in xs if x >= width * 0.67)
        
        top = sum(1 for y in ys if y < height * 0.33)
        middle = sum(1 for y in ys if height * 0.33 <= y < height * 0.67)
        bottom = sum(1 for y in ys if y >= height * 0.67)
        
        # Pattern detection
        x_std = np.std(xs)
        y_std = np.std(ys)
        
        if x_std < width * 0.1 and y_std < height * 0.1:
            pattern = "clustered together"
        elif x_std > width * 0.3 and y_std > height * 0.3:
            pattern = "scattered across the image"
        elif y_std < height * 0.15:
            pattern = "arranged in a horizontal line"
        elif x_std < width * 0.15:
            pattern = "arranged in a vertical line"
        else:
            pattern = "distributed across the image"
        
        # Build description
        region_desc = []
        if left > 0:
            region_desc.append(f"{left} on the left")
        if center > 0:
            region_desc.append(f"{center} in the center")
        if right > 0:
            region_desc.append(f"{right} on the right")
        
        desc = f"{len(dots)} red dots {pattern}"
        if region_desc:
            desc += f" ({', '.join(region_desc)})"
        
        return desc
    
    def _analyze_colors(self, image_rgb: np.ndarray) -> Dict[str, any]:
        """
        Analyze dominant colors
        """
        # Convert to LAB for perceptual color analysis
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        
        # Dominant color (mean)
        mean_color = np.mean(image_rgb, axis=(0, 1))
        
        # Check if mostly white/light background
        lightness = np.mean(lab[:, :, 0])
        is_light_bg = lightness > 200
        
        return {
            'mean_rgb': tuple(mean_color.astype(int)),
            'lightness': float(lightness),
            'is_light_background': is_light_bg
        }
    
    def _detect_shapes(self, image_rgb: np.ndarray) -> Dict[str, int]:
        """
        Detect geometric shapes
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=30, maxLineGap=10)
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                  param1=50, param2=30, minRadius=5, maxRadius=100)
        
        return {
            'lines': len(lines) if lines is not None else 0,
            'circles': len(circles[0]) if circles is not None else 0
        }
    
    def _vlm_analysis(self, image_path: str, cv_results: Dict) -> Dict[str, any]:
        """
        Use Gemini VLM for semantic understanding
        """
        print("  🧠 Running VLM analysis...")
        
        try:
            # Load image for Gemini
            pil_image = Image.open(image_path)
            
            # Create detailed prompt for VLM
            prompt = f"""Analyze this image in detail. I detected {cv_results['num_red_dots']} red dots using computer vision.

Please describe:
1. What objects or elements do you see in the image?
2. Where are the red dots located? Be spatially specific.
3. What is the background/context?
4. Are the dots part of any pattern or puzzle?
5. What action would make sense with these dots? (e.g., connecting, erasing, highlighting)

Provide a concise but detailed description."""
            
            response = self.vlm_model.generate_content([prompt, pil_image])
            vlm_description = response.text
            
            print(f"    VLM description: {vlm_description[:100]}...")
            
            return {
                'vlm_description': vlm_description,
                'vlm_available': True
            }
        
        except Exception as e:
            print(f"    ⚠ VLM analysis failed: {e}")
            return {
                'vlm_description': None,
                'vlm_available': False
            }
    
    def _generate_enhanced_prompt(self, analysis: Dict) -> str:
        """
        Generate prompt that incorporates visual understanding
        """
        # Base task (you can customize this)
        base_action = "erase the red dots with a white eraser"
        
        # Add spatial context
        spatial = analysis.get('spatial_distribution', '')
        num_dots = analysis.get('num_red_dots', 0)
        
        # Build enhanced prompt
        enhanced = f"{base_action}. "
        
        if num_dots > 0:
            enhanced += f"There are {num_dots} red dots {spatial}. "
        
        # Add VLM insights if available
        if analysis.get('vlm_available') and analysis.get('vlm_description'):
            vlm_desc = analysis['vlm_description']
            # Extract key phrases (simplified)
            enhanced += "Focus on each dot systematically. "
        
        # Add visual guidance
        if analysis.get('color_info', {}).get('is_light_background'):
            enhanced += "The eraser should clearly show white strokes on the light background. "
        
        enhanced += "Show smooth, natural erasing motion for each dot."
        
        return enhanced


def main():
    """
    Example usage
    """
    print("=" * 60)
    print("  Visual Recognition Pipeline for Red Dot Task")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = VisualRecognitionPipeline()
    
    # Analyze image
    image_path = "ltx_video_source/images/dot.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    analysis = pipeline.analyze_image(image_path)
    
    # Print results
    print("\n" + "=" * 60)
    print("  ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\n📍 Image: {analysis['image_path']}")
    print(f"📐 Size: {analysis['image_size']}")
    print(f"\n🔴 Red Dots Found: {analysis['num_red_dots']}")
    
    if analysis['num_red_dots'] > 0:
        print(f"📊 Distribution: {analysis['spatial_distribution']}")
        print(f"\nDot details:")
        for dot in analysis['red_dots'][:5]:  # Show first 5
            print(f"  • Dot {dot['id']}: center={dot['center']}, radius={dot['radius']}")
    
    if analysis.get('vlm_description'):
        print(f"\n🧠 VLM Understanding:")
        print(f"  {analysis['vlm_description']}")
    
    print(f"\n✨ ENHANCED PROMPT:")
    print(f"  {analysis['enhanced_prompt']}")
    
    print("\n" + "=" * 60)
    print("💡 Use this enhanced prompt for better generation!")
    print("=" * 60)


if __name__ == "__main__":
    main()

