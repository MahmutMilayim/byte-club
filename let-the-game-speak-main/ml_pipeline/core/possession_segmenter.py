"""
Possession Segmenter Module

Converts raw events (pass, dribble, shot) into high-level possession segments
with rich metadata for intelligent commentary generation.

Each possession segment includes:
- intent: build_up, counter, probe, overload, switch_play
- tempo: slow, medium, fast
- zone: own_half, mid_field, final_third, box_edge
- progress: backward, lateral, forward, deep_forward
- pressure: low, medium, high
- outcome: continues, shot, goal, loss, foul, corner
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math


class Intent(str, Enum):
    """Possession intent/style classification"""
    BUILD_UP = "build_up"           # Slow, patient, low risk
    COUNTER = "counter"             # Fast, direct, few passes
    PROBE = "probe"                 # Final third circulation
    OVERLOAD = "overload"           # Box edge pressure
    TRANSITION = "transition"       # Unclear, changing possession


class Tempo(str, Enum):
    """Pass tempo classification"""
    SLOW = "slow"       # > 1.5s avg between passes
    MEDIUM = "medium"   # 0.8s - 1.5s
    FAST = "fast"       # < 0.8s


class Zone(str, Enum):
    """Field zone classification"""
    OWN_HALF = "own_half"           # 0-35m (defending)
    MID_FIELD = "mid_field"         # 35-70m
    FINAL_THIRD = "final_third"     # 70-90m
    BOX_EDGE = "box_edge"           # 90-105m (attacking)


class Progress(str, Enum):
    """Ball progress direction"""
    BACKWARD = "backward"       # field_x decreasing significantly
    LATERAL = "lateral"         # field_x stable, field_y changing
    FORWARD = "forward"         # field_x increasing moderately
    DEEP_FORWARD = "deep_forward"  # field_x increasing rapidly toward goal


class Pressure(str, Enum):
    """Defensive pressure level"""
    LOW = "low"         # < 2 opponents nearby
    MEDIUM = "medium"   # 2-3 opponents
    HIGH = "high"       # 4+ opponents


class Outcome(str, Enum):
    """Possession outcome"""
    CONTINUES = "continues"     # Possession ongoing
    SHOT = "shot"               # Shot taken (not goal)
    GOAL = "goal"               # Goal scored!
    LOSS = "loss"               # Turnover
    FOUL = "foul"               # Foul won
    CORNER = "corner"           # Corner won
    OUT = "out"                 # Ball out of play


@dataclass
class PossessionSegment:
    """A single possession segment with rich metadata"""
    # Timing
    start_time: float           # seconds
    end_time: float             # seconds
    start_frame: int
    end_frame: int
    duration: float             # seconds
    
    # Team info
    team_id: str                # "L" or "R"
    team_name: str              # "Sol Takım" or "Sağ Takım"
    
    # Event counts
    pass_count: int
    dribble_count: int
    total_events: int
    
    # Classification
    intent: str                 # Intent enum value
    tempo: str                  # Tempo enum value
    zone: str                   # Primary zone (where most action happened)
    zone_start: str             # Starting zone
    zone_end: str               # Ending zone
    progress: str               # Progress enum value
    pressure: str               # Pressure enum value
    outcome: str                # Outcome enum value
    
    # Metrics
    avg_pass_time: float        # Average time between passes
    field_x_start: float        # Starting field x position
    field_x_end: float          # Ending field x position
    field_progress: float       # Net x progress (positive = attacking)
    pass_sequence: List[str]    # List of pass types: ["short", "long", "through"]
    
    # For commentary
    is_dangerous: bool          # High threat possession
    is_highlight: bool          # Worth emphasizing
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PossessionSegmenter:
    """
    Converts raw events into possession segments with intent classification.
    
    Usage:
        segmenter = PossessionSegmenter()
        segments = segmenter.segment(events, context)
    """
    
    # Field dimensions (standard: 105m x 68m)
    FIELD_LENGTH = 105.0
    FIELD_WIDTH = 68.0
    
    # Zone boundaries (in meters from defending goal)
    ZONE_OWN_END = 35.0
    ZONE_MID_END = 70.0
    ZONE_FINAL_END = 90.0
    
    # Tempo thresholds (seconds between passes)
    TEMPO_FAST_THRESHOLD = 0.8
    TEMPO_SLOW_THRESHOLD = 1.5
    
    # Progress thresholds (field_x change)
    PROGRESS_BACKWARD_THRESHOLD = -5.0
    PROGRESS_FORWARD_THRESHOLD = 5.0
    PROGRESS_DEEP_THRESHOLD = 15.0
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
    
    def segment(self, events: List[Dict], context: Dict) -> List[PossessionSegment]:
        """
        Convert raw events into possession segments.
        
        Args:
            events: List of raw events (pass, dribble, shot, etc.)
            context: {fps, team_left, team_right, frames_data, ...}
        
        Returns:
            List of PossessionSegment objects
        """
        if not events:
            return []
        
        self.fps = context.get("fps", 30.0)
        team_left = context.get("team_left", "Manchester United")
        team_right = context.get("team_right", "Bournemouth")
        frames_data = context.get("frames_data", [])
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.get("frame", x.get("start_time", 0)))
        
        # Group events into possession chains (same team continuous)
        possession_chains = self._group_by_possession(sorted_events)
        
        segments = []
        for chain in possession_chains:
            segment = self._analyze_chain(chain, team_left, team_right, frames_data)
            if segment:
                segments.append(segment)
        
        # Mark dangerous/highlight segments
        self._mark_highlights(segments)
        
        return segments
    
    # Maximum duration/events before splitting a possession
    MAX_SEGMENT_DURATION = 4.0  # seconds
    MAX_SEGMENT_EVENTS = 4      # events
    
    def _group_by_possession(self, events: List[Dict]) -> List[List[Dict]]:
        """
        Group consecutive events by the same team into possession chains.
        
        Also splits long possessions into smaller segments for better
        commentary granularity (max 4 seconds or 4 events per segment).
        """
        if not events:
            return []
        
        chains = []
        current_chain = []
        current_team = None
        chain_start_frame = 0
        
        for event in events:
            # Get team from event
            owner = event.get("start_owner") or event.get("player_id") or ""
            team = owner[0] if owner and owner[0] in ["L", "R"] else None
            
            # Check for possession break conditions
            event_type = event.get("type", event.get("segment_type", ""))
            is_turnover = event_type in ["loss", "turnover", "interception"]
            is_shot = event_type in ["shot", "shot_candidate"]
            
            # Get current frame for duration check
            current_frame = event.get("frame", event.get("start_time", 0))
            
            # Calculate chain duration so far
            chain_duration = (current_frame - chain_start_frame) / self.fps if chain_start_frame else 0
            
            # Check if we should split due to duration or event count
            should_split = (
                len(current_chain) >= self.MAX_SEGMENT_EVENTS or
                chain_duration >= self.MAX_SEGMENT_DURATION
            )
            
            # Turnover: önceki chain'e ekle ve sonra yeni chain başlat
            # Bu sayede önceki chain'in outcome'u "loss" olacak
            if is_turnover and current_chain:
                # Turnover'ı önceki chain'e ekle (outcome = loss için)
                current_chain.append(event)
                chains.append(current_chain)
                # Yeni chain başlat (turnover sonrası)
                current_chain = []
                current_team = None
                chain_start_frame = 0
            # Start new chain if team changes or segment too long
            elif team != current_team or (should_split and current_chain):
                if current_chain:
                    chains.append(current_chain)
                current_chain = [event]
                current_team = team
                chain_start_frame = current_frame
            else:
                current_chain.append(event)
                if not chain_start_frame:
                    chain_start_frame = current_frame
            
            # Shot ends a possession chain
            if is_shot and current_chain:
                chains.append(current_chain)
                current_chain = []
                current_team = None
                chain_start_frame = 0
        
        # Don't forget last chain
        if current_chain:
            chains.append(current_chain)
        
        return chains
    
    def _analyze_chain(self, chain: List[Dict], 
                       team_left: str, team_right: str,
                       frames_data: List[Dict]) -> Optional[PossessionSegment]:
        """Analyze a possession chain and create a segment with classifications."""
        if not chain:
            return None
        
        # Basic timing
        first_event = chain[0]
        last_event = chain[-1]
        
        start_frame = first_event.get("frame", first_event.get("start_time", 0))
        end_frame = last_event.get("frame", last_event.get("end_time", 
                                   last_event.get("start_time", start_frame)))
        
        # Handle end_frame for segments with duration
        # Check if end_time exists AND is not None
        if "end_time" in last_event and last_event["end_time"] is not None:
            end_frame = last_event["end_time"]
        elif "end_frame" in last_event and last_event["end_frame"] is not None:
            end_frame = last_event["end_frame"]
        
        # Ensure end_frame is not None (fallback to start_frame + 1)
        if end_frame is None:
            end_frame = start_frame + 1 if start_frame else 1
        
        start_time = start_frame / self.fps
        end_time = end_frame / self.fps
        duration = max(0.5, end_time - start_time)  # Minimum 0.5s
        
        # Team info
        owner = first_event.get("start_owner") or first_event.get("player_id") or "R1"
        team_id = owner[0] if owner and owner[0] in ["L", "R"] else "R"
        team_name = team_left if team_id == "L" else team_right
        
        # Count events by type
        pass_count = sum(1 for e in chain if e.get("type", e.get("segment_type")) == "pass")
        dribble_count = sum(1 for e in chain if e.get("type", e.get("segment_type")) == "dribble")
        total_events = len(chain)
        
        # Calculate tempo (average time between events)
        avg_pass_time = self._calculate_tempo(chain)
        tempo = self._classify_tempo(avg_pass_time)
        
        # Get field positions
        field_x_start, field_y_start = self._get_field_position(first_event, frames_data, prefer_end=False)
        field_x_end, field_y_end = self._get_field_position(last_event, frames_data, prefer_end=True)
        field_progress = field_x_end - field_x_start
        
        # For R team attacking left, invert progress direction
        # (negative field_progress = moving toward their goal = forward)
        if team_id == "R":
            field_progress = -field_progress
        
        # Classify zones (with team direction awareness)
        zone_start = self._classify_zone(field_x_start, team_id)
        zone_end = self._classify_zone(field_x_end, team_id)
        zone = self._get_primary_zone(chain, frames_data, team_id)
        
        # Classify progress direction
        progress = self._classify_progress(field_progress, field_y_start, field_y_end)
        
        # Classify pressure (from frames_data if available)
        pressure = self._classify_pressure(chain, frames_data)
        
        # Classify outcome
        outcome = self._classify_outcome(chain)
        
        # Classify intent based on all factors (including outcome for shot context)
        intent = self._classify_intent(
            pass_count=pass_count,
            tempo=tempo,
            zone=zone,
            progress=progress,
            field_progress=field_progress,
            duration=duration,
            outcome=outcome
        )
        
        # Build pass sequence
        pass_sequence = self._build_pass_sequence(chain)
        
        # Danger assessment
        is_dangerous = (
            zone in [Zone.FINAL_THIRD.value, Zone.BOX_EDGE.value] or
            outcome in [Outcome.SHOT.value, Outcome.GOAL.value] or
            (progress == Progress.DEEP_FORWARD.value and pass_count >= 2)
        )
        
        return PossessionSegment(
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            duration=duration,
            team_id=team_id,
            team_name=team_name,
            pass_count=pass_count,
            dribble_count=dribble_count,
            total_events=total_events,
            intent=intent,
            tempo=tempo,
            zone=zone,
            zone_start=zone_start,
            zone_end=zone_end,
            progress=progress,
            pressure=pressure,
            outcome=outcome,
            avg_pass_time=avg_pass_time,
            field_x_start=field_x_start,
            field_x_end=field_x_end,
            field_progress=field_progress,
            pass_sequence=pass_sequence,
            is_dangerous=is_dangerous,
            is_highlight=False  # Set later in _mark_highlights
        )
    
    def _safe_float(self, value, default: Optional[float] = None) -> Optional[float]:
        """Convert value to finite float, otherwise return default."""
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except (TypeError, ValueError):
            return default
    
    def _calculate_tempo(self, chain: List[Dict]) -> float:
        """Calculate average time between events in seconds."""
        if len(chain) < 2:
            return 1.0  # Default medium tempo
        
        times = []
        for i in range(1, len(chain)):
            prev_frame = chain[i-1].get("frame", chain[i-1].get("start_time", 0))
            curr_frame = chain[i].get("frame", chain[i].get("start_time", 0))
            time_diff = (curr_frame - prev_frame) / self.fps
            if time_diff > 0:
                times.append(time_diff)
        
        return sum(times) / len(times) if times else 1.0
    
    def _classify_tempo(self, avg_time: float) -> str:
        """Classify tempo based on average time between passes."""
        if avg_time < self.TEMPO_FAST_THRESHOLD:
            return Tempo.FAST.value
        elif avg_time > self.TEMPO_SLOW_THRESHOLD:
            return Tempo.SLOW.value
        return Tempo.MEDIUM.value
    
    def _get_field_position(self, event: Dict, frames_data: List[Dict], prefer_end: bool = False) -> Tuple[float, float]:
        """Get field position for an event (x, y in meters)."""
        # First try direct field coordinates in event (supports multiple schemas)
        if prefer_end:
            x_candidates = [
                event.get("field_x"),
                event.get("end_field_x"),
                event.get("start_field_x"),
            ]
            y_candidates = [
                event.get("field_y"),
                event.get("end_field_y"),
                event.get("start_field_y"),
            ]
        else:
            x_candidates = [
                event.get("field_x"),
                event.get("start_field_x"),
                event.get("end_field_x"),
            ]
            y_candidates = [
                event.get("field_y"),
                event.get("start_field_y"),
                event.get("end_field_y"),
            ]
        x = None
        for candidate in x_candidates:
            parsed = self._safe_float(candidate, None)
            if parsed is not None:
                x = parsed
                break
        if x is not None:
            y = None
            for candidate in y_candidates:
                parsed = self._safe_float(candidate, None)
                if parsed is not None:
                    y = parsed
                    break
            return x, (y if y is not None else 34.0)
        
        # Try to get from frames_data
        frame_idx_raw = event.get("frame", event.get("start_frame", event.get("start_time", 0)))
        try:
            frame_idx = int(frame_idx_raw)
        except (TypeError, ValueError):
            frame_idx = 0
        owner_id = event.get("start_owner") or event.get("player_id")
        
        for frame in frames_data:
            try:
                f_idx = int(frame.get("frame_idx", -1))
            except (TypeError, ValueError):
                f_idx = -1
            if f_idx == frame_idx:
                for track in frame.get("tracks", []):
                    if str(track.get("track_id")) == str(owner_id):
                        tx = self._safe_float(track.get("field_x"), None)
                        ty = self._safe_float(track.get("field_y"), None)
                        if tx is not None:
                            return tx, (ty if ty is not None else 34.0)
        
        # Default: center of field
        return 52.5, 34.0
    
    def _classify_zone(self, field_x: float, team_id: str = None) -> str:
        """
        Classify zone based on field x position and attacking direction.
        
        Coordinate system:
        - field_x: 0 = left goal, 105 = right goal
        - L team: defends left goal (x=0), attacks right goal (x=105)
        - R team: defends right goal (x=105), attacks left goal (x=0)
        
        Zone perspective is from the POSSESSING team's view:
        - "own_half" = near their own goal (defending zone)
        - "final_third" = near opponent's goal (attacking zone)
        
        Zone boundaries (attacking direction):
        - own_half: 0-35m from own goal
        - mid_field: 35-70m
        - final_third: 70-90m  
        - box_edge: 90-105m (near opponent's goal)
        """
        if team_id == "R":
            # R team attacks left (toward x=0), defends right (x=105)
            # Low field_x = near opponent's goal = attacking zone
            # High field_x = near own goal = defending zone
            distance_to_opponent_goal = field_x  # Distance to x=0
            distance_from_own_goal = self.FIELD_LENGTH - field_x  # Distance from x=105
        else:
            # L team attacks right (toward x=105), defends left (x=0)
            # High field_x = near opponent's goal = attacking zone
            # Low field_x = near own goal = defending zone
            distance_to_opponent_goal = self.FIELD_LENGTH - field_x  # Distance to x=105
            distance_from_own_goal = field_x  # Distance from x=0
        
        # Classify based on distance from own goal (how far we've progressed)
        if distance_from_own_goal < self.ZONE_OWN_END:  # 0-35m from own goal
            return Zone.OWN_HALF.value
        elif distance_from_own_goal < self.ZONE_MID_END:  # 35-70m from own goal
            return Zone.MID_FIELD.value
        elif distance_from_own_goal < self.ZONE_FINAL_END:  # 70-90m from own goal
            return Zone.FINAL_THIRD.value
        return Zone.BOX_EDGE.value  # 90-105m from own goal (near opponent's goal)
    
    def _get_primary_zone(self, chain: List[Dict], frames_data: List[Dict], team_id: str = None) -> str:
        """Get the zone where most of the possession occurred."""
        zone_counts = {z.value: 0 for z in Zone}
        
        for event in chain:
            x, _ = self._get_field_position(event, frames_data)
            zone = self._classify_zone(x, team_id)
            zone_counts[zone] += 1
        
        return max(zone_counts, key=zone_counts.get)
    
    def _classify_progress(self, field_x_change: float, 
                          y_start: float, y_end: float) -> str:
        """Classify ball progress direction."""
        if field_x_change < self.PROGRESS_BACKWARD_THRESHOLD:
            return Progress.BACKWARD.value
        elif field_x_change > self.PROGRESS_DEEP_THRESHOLD:
            return Progress.DEEP_FORWARD.value
        elif field_x_change > self.PROGRESS_FORWARD_THRESHOLD:
            return Progress.FORWARD.value
        return Progress.LATERAL.value
    
    def _classify_pressure(self, chain: List[Dict], 
                          frames_data: List[Dict]) -> str:
        """Classify defensive pressure level."""
        if not frames_data:
            return Pressure.MEDIUM.value
        
        # Sample a few frames from the chain
        opponent_counts = []
        
        for event in chain[:3]:  # Check first 3 events
            frame_idx = event.get("frame", event.get("start_time", 0))
            owner_id = event.get("start_owner") or event.get("player_id")
            
            for frame in frames_data:
                if frame.get("frame_idx") == frame_idx:
                    owner_track = None
                    for track in frame.get("tracks", []):
                        if track.get("track_id") == owner_id:
                            owner_track = track
                            break
                    
                    if owner_track:
                        owner_x = self._safe_float(owner_track.get("field_x"), None)
                        owner_y = self._safe_float(owner_track.get("field_y"), None)
                        if owner_x is None or owner_y is None:
                            continue
                        owner_team = owner_id[0] if owner_id else "R"
                        
                        opponents_nearby = 0
                        for track in frame.get("tracks", []):
                            track_id = track.get("track_id", "")
                            if track_id and track_id[0] != owner_team:
                                tx = self._safe_float(track.get("field_x"), None)
                                ty = self._safe_float(track.get("field_y"), None)
                                if tx is None or ty is None:
                                    continue
                                dist = math.sqrt((tx - owner_x)**2 + (ty - owner_y)**2)
                                if dist < 10:  # Within 10 meters
                                    opponents_nearby += 1
                        
                        opponent_counts.append(opponents_nearby)
        
        if not opponent_counts:
            return Pressure.MEDIUM.value
        
        avg_opponents = sum(opponent_counts) / len(opponent_counts)
        
        if avg_opponents < 2:
            return Pressure.LOW.value
        elif avg_opponents >= 4:
            return Pressure.HIGH.value
        return Pressure.MEDIUM.value
    
    def _classify_outcome(self, chain: List[Dict]) -> str:
        """Classify possession outcome."""
        if not chain:
            return Outcome.CONTINUES.value
        
        last_event = chain[-1]
        event_type = last_event.get("type", last_event.get("segment_type", ""))
        
        # Check for goal
        if last_event.get("goal") or last_event.get("is_goal"):
            return Outcome.GOAL.value
        
        # Check event type
        if event_type in ["shot", "shot_candidate"]:
            return Outcome.SHOT.value
        elif event_type in ["loss", "turnover", "interception"]:
            return Outcome.LOSS.value
        elif event_type == "foul":
            return Outcome.FOUL.value
        elif event_type == "corner":
            return Outcome.CORNER.value
        
        return Outcome.CONTINUES.value
    
    def _classify_intent(self, pass_count: int, tempo: str, zone: str,
                        progress: str, field_progress: float, 
                        duration: float, outcome: str = None) -> str:
        """
        Classify possession intent based on multiple factors.
        
        Intent types:
        - BUILD_UP: Slow, patient, from own half
        - COUNTER: Fast, direct, few passes, big progress
        - PROBE: Final third circulation, looking for opening
        - OVERLOAD: Box edge pressure, multiple attempts
        """
        # If outcome is shot or goal, this is an attacking play - use probe/overload
        if outcome in [Outcome.SHOT.value, Outcome.GOAL.value]:
            if zone == Zone.BOX_EDGE.value or pass_count >= 3:
                return Intent.OVERLOAD.value
            return Intent.PROBE.value
        
        # Counter attack: fast tempo, big progress, few passes
        if (tempo == Tempo.FAST.value and 
            field_progress > 20 and 
            pass_count <= 3 and
            duration < 4.0):
            return Intent.COUNTER.value
        
        # Probe: final third, medium/slow tempo, looking for opening
        if zone in [Zone.FINAL_THIRD.value, Zone.BOX_EDGE.value]:
            if pass_count >= 2 and tempo != Tempo.FAST.value:
                return Intent.PROBE.value
            elif pass_count >= 3:
                return Intent.OVERLOAD.value
            # Even with 1 pass, if in final third, it's probe not build_up
            return Intent.PROBE.value
        
        # Build up: slow/medium tempo, own half or mid, patient
        if (tempo in [Tempo.SLOW.value, Tempo.MEDIUM.value] and
            zone in [Zone.OWN_HALF.value, Zone.MID_FIELD.value]):
            return Intent.BUILD_UP.value
        
        # Default: transition
        return Intent.TRANSITION.value
    
    def _build_pass_sequence(self, chain: List[Dict]) -> List[str]:
        """Build a list of pass types in the chain."""
        sequence = []
        
        for event in chain:
            event_type = event.get("type", event.get("segment_type", ""))
            
            if event_type == "pass":
                # Classify pass type
                displacement = event.get("displacement", 0)
                speed = event.get("average_speed", 0)
                
                if displacement > 25 or speed > 15:
                    sequence.append("long")
                elif displacement < 10:
                    sequence.append("short")
                else:
                    sequence.append("medium")
            
            elif event_type == "dribble":
                sequence.append("dribble")
            
            elif event_type in ["shot", "shot_candidate"]:
                sequence.append("shot")
        
        return sequence
    
    def _mark_highlights(self, segments: List[PossessionSegment]) -> None:
        """Mark segments that are highlights (worth emphasizing)."""
        for segment in segments:
            segment.is_highlight = (
                segment.outcome == Outcome.GOAL.value or
                (segment.outcome == Outcome.SHOT.value and segment.zone == Zone.BOX_EDGE.value) or
                (segment.intent == Intent.COUNTER.value and segment.is_dangerous) or
                (segment.pass_count >= 5 and segment.progress == Progress.DEEP_FORWARD.value)
            )


def segment_possessions(events: List[Dict], context: Dict) -> List[Dict]:
    """
    Convenience function to segment events into possessions.
    
    Returns list of dicts (for JSON serialization).
    """
    segmenter = PossessionSegmenter(fps=context.get("fps", 30.0))
    segments = segmenter.segment(events, context)
    return [seg.to_dict() for seg in segments]


# For testing
if __name__ == "__main__":
    import json
    
    # Test with sample events
    test_events = [
        {"type": "pass", "frame": 10, "start_owner": "R1", "end_owner": "R2", 
         "displacement": 15, "average_speed": 8},
        {"type": "dribble", "frame": 40, "start_owner": "R2", "end_owner": "R2",
         "displacement": 10, "average_speed": 3},
        {"type": "pass", "frame": 70, "start_owner": "R2", "end_owner": "R3",
         "displacement": 25, "average_speed": 12},
        {"type": "shot_candidate", "frame": 100, "start_owner": "R3", 
         "goal": False, "average_speed": 25}
    ]
    
    context = {
        "fps": 30.0,
        "team_left": "Manchester United",
        "team_right": "Bournemouth",
        "frames_data": []
    }
    
    segments = segment_possessions(test_events, context)
    print(json.dumps(segments, indent=2, ensure_ascii=False))
