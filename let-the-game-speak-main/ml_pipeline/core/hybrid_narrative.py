"""
Hybrid Commentary Generator

This module provides a unified interface for the new template-based
commentary system while maintaining backward compatibility with the
existing narrative pipeline.

The new system:
1. Segments raw events into possession-based segments
2. Classifies each segment (intent, zone, tempo, pressure, outcome)
3. Selects candidate templates (rule-based)
4. Uses LLM as editor (select + slot fill + flow)

Benefits:
- More consistent commentary
- Better duration control (templates have known TTS durations)
- Lower API costs (LLM as editor, not writer)
- Reduced hallucination
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

from .possession_segmenter import PossessionSegmenter, PossessionSegment, segment_possessions
from .template_commentary import TemplateCommentaryGenerator, TemplateBank, TemplateSelector


@dataclass
class HybridCommentary:
    """Commentary with timing and metadata"""
    text: str
    start_time: float
    end_time: float
    duration: float
    event_type: str
    event_frame: int
    segment_info: Dict
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "event_type": self.event_type,
            "event_frame": self.event_frame,
            "segment_info": self.segment_info
        }


class HybridNarrativeGenerator:
    """
    New template-based narrative generator.
    
    Replaces the old single-LLM approach with:
    1. Possession segmentation (context preservation)
    2. Template selection (rule-based, predictable)
    3. LLM editing (slot filling, flow, minimal hallucination)
    
    Usage:
        generator = HybridNarrativeGenerator()
        commentaries = generator.generate(events, context)
    """
    
    def __init__(self, 
                 templates_path: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the hybrid generator.
        
        Args:
            templates_path: Path to templates.json (optional)
            api_key: Deprecated parameter, kept for backward compatibility
        """
        self.segmenter = None  # Initialized with fps on first use
        self.generator = TemplateCommentaryGenerator(
            templates_path=templates_path
        )
        
        print(f"🎙️  Hybrid Narrative Generator initialized")
    
    def generate(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Generate timed commentary from raw events.
        
        Args:
            events: List of raw events (pass, dribble, shot, etc.)
            context: {
                fps: float,
                duration: float (video duration),
                team_left: str,
                team_right: str,
                player_names: Dict[str, str],
                frames_data: List[Dict] (optional)
            }
        
        Returns:
            List of commentary dicts with timing
        """
        fps = context.get("fps", 30.0)
        video_duration = context.get("duration", 10.0)
        
        # Step 1: Segment events into possessions
        print("📊 Step 1: Segmenting possessions...")
        self.segmenter = PossessionSegmenter(fps=fps)
        segments = self.segmenter.segment(events, context)
        
        print(f"   Found {len(segments)} possession segments")
        for i, seg in enumerate(segments):
            print(f"   [{i+1}] {seg.intent} | {seg.zone} | {seg.outcome} | "
                  f"{seg.pass_count}p | {seg.duration:.1f}s")
        
        # Step 2: Generate commentary using smart placement
        # - İlk yorum 0.1'de başlar
        # - Son yorum videonun sonuna yerleştirilir
        # - Aradaki yorumlar çakışmadan dağıtılır
        print("\n🎤 Step 2: Generating smart-placed commentary...")
        commentaries = self.generator.generate_for_segments_smart(segments, video_duration, context)
        
        print(f"   Generated {len(commentaries)} commentaries")
        for i, comm in enumerate(commentaries):
            text_preview = comm["text"][:50] + "..." if len(comm["text"]) > 50 else comm["text"]
            print(f"   [{i+1}] ({comm['start_time']:.1f}s) {text_preview}")
        
        return commentaries
    
    def generate_with_summary(self, events: List[Dict], context: Dict) -> Dict:
        """
        Generate commentary with summary statistics.
        
        Returns:
            {
                "narrative": str (combined text),
                "timed_segments": List[Dict],
                "possession_segments": List[Dict],
                "stats": {
                    "total_segments": int,
                    "passes": int,
                    "shots": int,
                    "goals": int,
                    "video_duration": float
                }
            }
        """
        # Generate commentaries
        commentaries = self.generate(events, context)
        
        # Build summary
        fps = context.get("fps", 30.0)
        self.segmenter = PossessionSegmenter(fps=fps)
        segments = self.segmenter.segment(events, context)
        
        # Count statistics
        total_passes = sum(seg.pass_count for seg in segments)
        total_shots = sum(1 for seg in segments if seg.outcome in ["shot", "goal"])
        total_goals = sum(1 for seg in segments if seg.outcome == "goal")
        
        # Combined narrative text
        narrative_text = " ".join(comm["text"] for comm in commentaries)
        
        # Convert segments to dicts
        segment_dicts = [seg.to_dict() for seg in segments]
        
        return {
            "narrative": narrative_text,
            "timed_segments": commentaries,
            "possession_segments": segment_dicts,
            "stats": {
                "total_segments": len(segments),
                "passes": total_passes,
                "shots": total_shots,
                "goals": total_goals,
                "video_duration": context.get("duration", 0)
            }
        }
    
    def reset(self):
        """Reset state for new video."""
        self.generator.reset()


def generate_hybrid_narrative(events: List[Dict], context: Dict) -> Dict:
    """
    Convenience function for generating hybrid narrative.
    
    Returns full result with narrative, timed_segments, and stats.
    """
    generator = HybridNarrativeGenerator()
    return generator.generate_with_summary(events, context)


# For testing
if __name__ == "__main__":
    # Test with sample events
    test_events = [
        {"type": "pass", "frame": 10, "start_time": 10, "end_time": 40,
         "start_owner": "R1", "end_owner": "R2", "displacement": 15, "average_speed": 8},
        {"type": "dribble", "frame": 50, "start_time": 50, "end_time": 80,
         "start_owner": "R2", "end_owner": "R2", "displacement": 10, "average_speed": 3},
        {"type": "pass", "frame": 90, "start_time": 90, "end_time": 120,
         "start_owner": "R2", "end_owner": "R3", "displacement": 25, "average_speed": 12},
        {"type": "shot_candidate", "frame": 130, "start_time": 130, "end_time": 150,
         "start_owner": "R3", "goal": False, "average_speed": 25}
    ]
    
    context = {
        "fps": 30.0,
        "duration": 5.0,
        "team_left": "Manchester United",
        "team_right": "Bournemouth",
        "frames_data": []
    }
    
    result = generate_hybrid_narrative(test_events, context)
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
