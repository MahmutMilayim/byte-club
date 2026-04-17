"""
Template-Based Commentary Generator

This module implements a hybrid template + LLM system for generating
natural football commentary:

1. TemplateSelector: Rule-based selection of 2-4 candidate templates
2. CommentaryEditor: LLM-based template selection, slot filling, and flow

The LLM acts as an EDITOR, not a writer - reducing hallucination and
ensuring duration constraints.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from .possession_segmenter import PossessionSegment, Intent, Zone, Outcome, Tempo
from .gemini_client import MODEL_ANALYSIS, generate_text, setup

load_dotenv()


@dataclass
class TemplateCandidate:
    """A candidate template for commentary"""
    text: str
    duration: float
    tone: str
    intent: str
    zone: str
    outcome: str
    duration_range: str  # "short", "medium", "long"
    id: str = ""  # Unique template ID to prevent duplicates
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "duration": self.duration,
            "tone": self.tone,
            "id": self.id
        }


class TemplateBank:
    """
    Loads and manages the multi-layer template bank.
    
    Template hierarchy:
    intent -> zone -> outcome -> duration_range -> [templates]
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        if templates_path is None:
            # Default path relative to this file
            templates_path = Path(__file__).parent.parent.parent / "commentary_templates" / "templates.json"
        
        self.templates_path = Path(templates_path)
        self.templates = self._load_templates()
        self.connectors = self.templates.get("connectors", {})
        self.outcomes = self.templates.get("outcomes", {})
        self._validate_template_ids()
    
    def _validate_template_ids(self) -> None:
        """Validate that all templates have unique IDs."""
        all_ids = set()
        missing_id_count = 0
        duplicate_ids = []
        
        def check_templates(data, path=""):
            nonlocal missing_id_count
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "text" in item:
                        template_id = item.get("id", "")
                        if not template_id:
                            missing_id_count += 1
                            print(f"⚠️  Missing ID in template: '{item['text'][:40]}...' at {path}[{i}]")
                        elif template_id in all_ids:
                            duplicate_ids.append((template_id, path))
                        else:
                            all_ids.add(template_id)
                    elif isinstance(item, dict):
                        check_templates(item, f"{path}[{i}]")
            elif isinstance(data, dict):
                for key, value in data.items():
                    if key.startswith("_"):  # Skip meta fields
                        continue
                    check_templates(value, f"{path}.{key}" if path else key)
        
        check_templates(self.templates)
        
        if missing_id_count > 0:
            print(f"⚠️  Total templates without ID: {missing_id_count}")
        if duplicate_ids:
            print(f"❌ Duplicate IDs found: {duplicate_ids}")
        if missing_id_count == 0 and not duplicate_ids:
            print(f"✅ All {len(all_ids)} templates have unique IDs")
    
    def get_average_template_duration(self) -> float:
        """
        Calculate average duration of all templates in the bank.
        Falls back to 2.5s if no templates found.
        """
        durations = []
        
        def collect_durations(data):
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "duration" in item:
                        durations.append(item["duration"])
                    elif isinstance(item, dict):
                        collect_durations(item)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if key.startswith("_"):  # Skip meta fields
                        continue
                    collect_durations(value)
        
        collect_durations(self.templates)
        
        if durations:
            avg = sum(durations) / len(durations)
            print(f"📊 Template bank: {len(durations)} templates, avg duration: {avg:.2f}s")
            return avg
        return 2.5  # Default fallback
    
    def _load_templates(self) -> Dict:
        """Load templates from JSON file."""
        if not self.templates_path.exists():
            print(f"⚠️  Template file not found: {self.templates_path}")
            return {}
        
        with open(self.templates_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_templates(self, intent: str, zone: str, outcome: str, 
                     duration_range: str) -> List[Dict]:
        """
        Get templates matching the given criteria.
        
        Falls back to more general templates if specific ones not found.
        """
        templates = []
        
        # Shot outcomes artık short template'lere sahip - dönüşüm yapmıyoruz
        is_shot_outcome = outcome in ["shot", "shot_saved", "shot_wide", "shot_post"]
        
        # Try exact match first
        intent_data = self.templates.get(intent, {})
        zone_data = intent_data.get(zone, intent_data.get("_default", {}))
        outcome_data = zone_data.get(outcome, zone_data.get("continues", {}))
        duration_templates = outcome_data.get(duration_range, [])
        
        if duration_templates:
            for t in duration_templates:
                templates.append({
                    **t,
                    "intent": intent,
                    "zone": zone,
                    "outcome": outcome,
                    "duration_range": duration_range
                })
        
        # Fallback: try adjacent duration ranges (skip "short" for shot outcomes)
        if not templates:
            fallback_durations = ["medium", "long"] if is_shot_outcome else ["medium", "short", "long"]
            for fallback_dur in fallback_durations:
                if fallback_dur != duration_range:
                    fallback_templates = outcome_data.get(fallback_dur, [])
                    if fallback_templates:
                        for t in fallback_templates:
                            templates.append({
                                **t,
                                "intent": intent,
                                "zone": zone,
                                "outcome": outcome,
                                "duration_range": fallback_dur
                            })
                        break
        
        # Shot outcomes için özel fallback: outcomes bölümünden al
        if not templates and is_shot_outcome:
            outcome_type = "shot_saved"  # shot → shot_saved template'leri
            outcome_templates = self.get_outcome_templates(outcome_type, duration_range)
            for t in outcome_templates:
                templates.append({
                    **t,
                    "intent": intent,
                    "zone": zone,
                    "outcome": outcome,
                    "duration_range": duration_range
                })
        
        # Fallback: try "continues" outcome (sadece shot değilse)
        if not templates and outcome != "continues" and not is_shot_outcome:
            continues_data = zone_data.get("continues", {})
            for dur in [duration_range, "medium", "short"]:
                fallback_templates = continues_data.get(dur, [])
                if fallback_templates:
                    for t in fallback_templates:
                        templates.append({
                            **t,
                            "intent": intent,
                            "zone": zone,
                            "outcome": "continues",
                            "duration_range": dur
                        })
                    break
        
        # Fallback: transition default (sadece shot değilse)
        if not templates and not is_shot_outcome:
            transition_data = self.templates.get("transition", {}).get("_default", {})
            continues_data = transition_data.get("continues", {})
            for dur in [duration_range, "medium", "short"]:
                fallback_templates = continues_data.get(dur, [])
                if fallback_templates:
                    for t in fallback_templates:
                        templates.append({
                            **t,
                            "intent": "transition",
                            "zone": "_default",
                            "outcome": "continues",
                            "duration_range": dur
                        })
                    break
        
        return templates
    
    def get_outcome_templates(self, outcome_type: str, 
                              duration_range: str) -> List[Dict]:
        """Get specific outcome templates (goal, shot_saved, etc.)"""
        outcome_data = self.outcomes.get(outcome_type, {})
        
        # Artık short template'ler var - dönüşüm yapmıyoruz
        templates = outcome_data.get(duration_range, [])
        
        if not templates:
            # Fallback to other durations - short yoksa medium dene
            fallback_durations = ["medium", "long"] if duration_range == "short" else ["short", "medium", "long"]
            for dur in fallback_durations:
                if dur != duration_range:
                    templates = outcome_data.get(dur, [])
                    if templates:
                        break
        
        return [{"text": t["text"], "duration": t["duration"], 
                 "tone": t.get("tone", "neutral"), "id": t.get("id", "")} for t in templates]
    
    def get_connector(self, context_type: str, used_ids: set = None) -> Optional[Dict]:
        """Get a connector phrase for flow between segments.
        
        Args:
            context_type: Type of connector to get
            used_ids: Set of already used template IDs to exclude
            
        Returns:
            Dict with text and id, or None if no connector available
        """
        connectors = self.connectors.get(context_type, [])
        if not connectors:
            return None
        
        # Filter out used connectors if used_ids provided
        if used_ids:
            available = [c for c in connectors if isinstance(c, dict) and c.get("id", "") not in used_ids]
            if available:
                connector = random.choice(available)
                return {"text": connector.get("text", connector) if isinstance(connector, dict) else connector,
                        "id": connector.get("id", "") if isinstance(connector, dict) else ""}
        
        # Fallback: return any connector
        connector = random.choice(connectors)
        if isinstance(connector, dict):
            return {"text": connector.get("text", ""), "id": connector.get("id", "")}
        # Legacy support for string connectors
        return {"text": connector, "id": ""}

    def get_filler(self, max_duration: float, used_ids: set = None) -> Optional[Dict]:
        """Get a short filler comment for gaps between commentaries.
        
        Args:
            max_duration: Maximum duration for the filler
            used_ids: Set of already used template IDs to exclude
            
        Returns:
            Dict with text, duration, tone, and id, or None if no filler available
        """
        fillers_data = self.templates.get("fillers", {})
        
        # Combine short and medium fillers
        all_fillers = []
        for duration_type in ["short", "medium"]:
            fillers = fillers_data.get(duration_type, [])
            all_fillers.extend(fillers)
        
        if not all_fillers:
            return None
        
        # Filter by max_duration
        fitting_fillers = [f for f in all_fillers if f.get("duration", 2.0) <= max_duration]
        
        if not fitting_fillers:
            return None
        
        # Filter out used fillers if used_ids provided
        if used_ids:
            available = [f for f in fitting_fillers if f.get("id", "") not in used_ids]
            if available:
                filler = random.choice(available)
            else:
                # All used, pick any fitting one
                filler = random.choice(fitting_fillers)
        else:
            filler = random.choice(fitting_fillers)
        
        return {
            "text": filler.get("text", ""),
            "duration": filler.get("duration", 1.5),
            "tone": filler.get("tone", "neutral"),
            "id": filler.get("id", "")
        }


class TemplateSelector:
    """
    Rule-based selection of candidate templates based on possession segment labels.
    
    Selects 2-4 candidates that match:
    - Intent (build_up, counter, probe, etc.)
    - Zone (own_half, mid_field, final_third, box_edge)
    - Outcome (continues, shot, goal, loss, etc.)
    - Duration range (based on segment duration)
    """
    
    # Duration range thresholds (seconds)
    SHORT_MAX = 2.0
    MEDIUM_MAX = 3.5
    
    def __init__(self, template_bank: TemplateBank):
        self.bank = template_bank
    
    def select_candidates(self, segment: PossessionSegment, 
                         target_duration: Optional[float] = None,
                         used_ids: Optional[set] = None,
                         prefer_no_slots: bool = False,
                         prefer_longer: bool = False,
                         prefer_short: bool = False,
                         max_duration: Optional[float] = None) -> List[TemplateCandidate]:
        """
        Select 2-4 candidate templates for a possession segment.
        
        Args:
            segment: PossessionSegment with intent, zone, outcome, etc.
            target_duration: Target commentary duration (defaults to segment duration)
            used_ids: Set of already used template IDs to exclude
            prefer_no_slots: If True, prefer templates without {team} or other slots
            prefer_longer: If True, prefer longer duration templates
            prefer_short: If True, prefer shorter duration templates (for shot/goal at video end)
            max_duration: Maximum allowed duration for template (for video end constraints)
        
        Returns:
            List of TemplateCandidate objects
        """
        duration = target_duration or segment.duration
        
        # =======================================================================
        # ŞUT/GOL İÇİN ZONE DÜZELTMESİ
        # =======================================================================
        # Own half veya mid_field'dan şut olamaz (mantıken yanlış tespit)
        # Bu durumda zone'u final_third'e düzelt ki doğru template seçilsin
        # =======================================================================
        effective_zone = segment.zone
        if segment.outcome in ["goal", "shot", "shot_saved", "shot_wide", "shot_post"]:
            if segment.zone in ["own_half", "mid_field"]:
                effective_zone = "final_third"
                print(f"   🔧 Şut/Gol zone düzeltmesi: {segment.zone} → {effective_zone}")
        
        # Şut/gol için: max_duration'a göre akıllı duration_range seç
        # Artık prefer_short kullanılmıyor - kalan süreye göre en uygun template seçilir
        if max_duration is not None:
            # Kalan süreye göre duration range belirle
            if max_duration >= 4.0:
                duration_range = "long"  # Bolca süre var, uzun template seçilebilir
            elif max_duration >= 2.5:
                duration_range = "medium"  # Orta süre, medium tercih
            elif max_duration >= 1.5:
                duration_range = "short"  # Az süre, kısa template
            else:
                duration_range = "short"  # Çok az süre, en kısa
            print(f"   📏 Kalan süre: {max_duration:.1f}s → duration_range: {duration_range}")
        elif prefer_longer:
            duration_range = "long"
        elif prefer_short:
            duration_range = "short"
        else:
            duration_range = self._classify_duration(duration)
        
        # Shot outcomes (not goal): "short" yerine "medium" kullan (daha iyi template'ler var)
        # Ama max_duration çok kısaysa (< 2s) short kullanılabilir
        if segment.outcome in ["shot", "shot_saved", "shot_wide", "shot_post"]:
            if duration_range == "short" and (max_duration is None or max_duration >= 2.0):
                duration_range = "medium"
        
        used_ids = used_ids or set()
        
        # Helper to filter out used templates
        def filter_unused(templates):
            return [t for t in templates if t.get("id", "") not in used_ids]
        
        # Helper to filter by max duration (for video end constraints)
        def filter_by_max_duration(templates):
            if max_duration is None:
                return templates
            # Filter templates that fit within max_duration (with 0.3s tolerance for TTS variance)
            # 0.5s çok fazlaydı, 0.3s daha güvenli
            fitting = [t for t in templates if t.get("duration", 2.0) <= max_duration + 0.3]
            if fitting:
                # Sort by duration descending - prefer longer templates that still fit
                return sorted(fitting, key=lambda t: t.get("duration", 2.0), reverse=True)
            # If nothing fits, return empty - don't return templates that exceed video duration
            # Bu sayede sistem daha kısa duration_range deneyecek
            return []
        
        # Helper to prefer templates without slots like {team}
        def sort_by_slot_preference(templates):
            if not prefer_no_slots:
                return templates
            # Sort: templates without {team} first
            no_slots = [t for t in templates if "{team}" not in t.get("text", "")]
            with_slots = [t for t in templates if "{team}" in t.get("text", "")]
            return no_slots + with_slots
        
        # =======================================================================
        # PAS SAYISINA GÖRE AKILLI FİLTRELEME
        # =======================================================================
        # Eğer segment'te az pas varsa (≤2), çoklu pas ima eden template'leri filtrele
        # Bu sayede "dar alanda paslaşmalar" gibi yorumlar tek pas için söylenmez
        # =======================================================================
        MULTI_PASS_KEYWORDS = [
            "paslaşmalar",      # çoğul - birden fazla pas
            "seri pas",         # seri = ardışık
            "pas trafiği",      # trafik = yoğunluk
            "üçgenler",         # üçgen pas = en az 3 oyuncu
            "dar alanda",       # dar alan = sıkı paslaşma
            "topu döndür",      # döndürmek = uzun süreli
            "top döndür",
            "sabırla top",
            "sağlı sollu",      # sağ-sol = çoklu pas
            "birkaç pas",
            "kısa paslaşma",    # paslaşma = çoğul
        ]
        
        MIN_PASSES_FOR_MULTI_PASS_TEMPLATES = 3  # Bu sayıdan az pas varsa filtrele
        
        def filter_by_pass_count(templates):
            """Pas sayısı düşükse çoklu pas ima eden template'leri filtrele"""
            pass_count = segment.pass_count
            
            # Yeterli pas varsa filtreleme yapma
            if pass_count >= MIN_PASSES_FOR_MULTI_PASS_TEMPLATES:
                return templates
            
            # Az pas varsa, çoklu pas ima eden template'leri çıkar
            filtered = []
            for t in templates:
                text_lower = t.get("text", "").lower()
                has_multi_pass_keyword = any(kw in text_lower for kw in MULTI_PASS_KEYWORDS)
                
                if not has_multi_pass_keyword:
                    filtered.append(t)
            
            # Eğer tüm template'ler filtrelendiyse, en azından birini döndür
            if not filtered and templates:
                return templates[:1]
            
            return filtered
        
        # Get primary candidates (şimdi pas sayısı filtrelemesi de var)
        primary_templates = sort_by_slot_preference(filter_by_max_duration(filter_by_pass_count(filter_unused(self.bank.get_templates(
            intent=segment.intent,
            zone=effective_zone,  # Şut/gol için düzeltilmiş zone
            outcome=segment.outcome,
            duration_range=duration_range
        )))))
        
        # FALLBACK: Eğer hiç template bulunamadıysa, diğer duration_range'leri dene
        # Shot/goal için short yoksa medium'a da bak (yukarı fallback)
        if not primary_templates and max_duration is not None:
            if duration_range == "long":
                fallback_ranges = ["medium", "short"]
            elif duration_range == "medium":
                fallback_ranges = ["short"]
            elif duration_range == "short":
                # Short template yoksa medium'a fallback yap (özellikle shot/goal için)
                fallback_ranges = ["medium"]
            else:
                fallback_ranges = []
            for fallback_range in fallback_ranges:
                primary_templates = sort_by_slot_preference(filter_by_max_duration(filter_by_pass_count(filter_unused(self.bank.get_templates(
                    intent=segment.intent,
                    zone=effective_zone,  # Şut/gol için düzeltilmiş zone
                    outcome=segment.outcome,
                    duration_range=fallback_range
                )))))
                if primary_templates:
                    print(f"   🔄 Fallback: {duration_range} → {fallback_range} duration_range'e düşüldü")
                    duration_range = fallback_range
                    break
        
        candidates = []
        
        # Add primary candidates (up to 2)
        for t in primary_templates[:2]:
            candidates.append(TemplateCandidate(
                text=t["text"],
                duration=t.get("duration", 2.0),
                tone=t.get("tone", "neutral"),
                intent=t.get("intent", segment.intent),
                zone=t.get("zone", segment.zone),
                outcome=t.get("outcome", segment.outcome),
                duration_range=t.get("duration_range", duration_range),
                id=t.get("id", "")
            ))
        
        # Add secondary candidates (different zone or intent for variety)
        if len(candidates) < 3:
            # Try adjacent zone
            adjacent_zone = self._get_adjacent_zone(effective_zone, segment.progress)
            if adjacent_zone:
                alt_templates = filter_by_pass_count(filter_unused(self.bank.get_templates(
                    intent=segment.intent,
                    zone=adjacent_zone,
                    outcome=segment.outcome,
                    duration_range=duration_range
                )))
                for t in alt_templates[:1]:
                    candidates.append(TemplateCandidate(
                        text=t["text"],
                        duration=t.get("duration", 2.0),
                        tone=t.get("tone", "neutral"),
                        intent=t.get("intent", segment.intent),
                        zone=t.get("zone", adjacent_zone),
                        outcome=t.get("outcome", segment.outcome),
                        duration_range=t.get("duration_range", duration_range),
                        id=t.get("id", "")
                    ))
        
        # CROSS-INTENT FALLBACK: If we don't have enough candidates, try similar intents
        if len(candidates) < 2:
            similar_intents = self._get_similar_intents(segment.intent)
            for alt_intent in similar_intents:
                if len(candidates) >= 3:
                    break
                cross_templates = filter_by_pass_count(filter_unused(self.bank.get_templates(
                    intent=alt_intent,
                    zone=effective_zone,  # Şut/gol için düzeltilmiş zone
                    outcome=segment.outcome,
                    duration_range=duration_range
                )))
                for t in cross_templates[:2]:
                    if len(candidates) >= 3:
                        break
                    candidates.append(TemplateCandidate(
                        text=t["text"],
                        duration=t.get("duration", 2.0),
                        tone=t.get("tone", "neutral"),
                        intent=alt_intent,
                        zone=t.get("zone", effective_zone),
                        outcome=t.get("outcome", segment.outcome),
                        duration_range=t.get("duration_range", duration_range),
                        id=t.get("id", "")
                    ))
        
        # For special outcomes (goal, shot, loss), add outcome-specific templates FIRST (priority)
        if segment.outcome in ["goal", "shot", "loss"]:
            if segment.outcome == "goal":
                outcome_type = "goal"
            elif segment.outcome == "shot":
                outcome_type = "shot_saved"
            else:
                outcome_type = "loss"
            outcome_templates = filter_by_max_duration(filter_unused(self.bank.get_outcome_templates(outcome_type, duration_range)))
            # Insert at beginning for priority
            for t in reversed(outcome_templates[:2]):
                candidates.insert(0, TemplateCandidate(
                    text=t["text"],
                    duration=t.get("duration", 1.5),
                    tone=t.get("tone", "excited" if outcome_type != "loss" else "critical"),
                    intent=segment.intent,
                    zone=segment.zone,
                    outcome=segment.outcome,
                    duration_range=duration_range,
                    id=t.get("id", "")
                ))
        
        # Ensure at least 2 candidates
        if len(candidates) < 2:
            # Add generic transition templates
            fallback = filter_unused(self.bank.get_templates(
                intent="transition",
                zone="_default",
                outcome="continues",
                duration_range=duration_range
            ))
            for t in fallback:
                if len(candidates) >= 2:
                    break
                candidates.append(TemplateCandidate(
                    text=t["text"],
                    duration=t.get("duration", 1.5),
                    tone=t.get("tone", "neutral"),
                    intent="transition",
                    zone="_default",
                    outcome="continues",
                    duration_range=duration_range,
                    id=t.get("id", "")
                ))
        
        # Warn about templates without IDs (they can cause overload issues)
        for candidate in candidates:
            if not candidate.id:
                print(f"⚠️  Template without ID: '{candidate.text[:30]}...' - this may cause overload issues")
        
        return candidates[:4]  # Max 4 candidates
    
    def _get_similar_intents(self, intent: str) -> List[str]:
        """Get similar intents for cross-intent fallback."""
        # Intent similarity mapping
        similarity_map = {
            "build_up": ["probe", "overload", "transition"],
            "probe": ["build_up", "overload", "transition"],
            "counter": ["transition", "probe"],
            "overload": ["probe", "build_up", "transition"],
            "transition": ["build_up", "probe"]
        }
        return similarity_map.get(intent, ["transition"])
    
    def _classify_duration(self, duration: float) -> str:
        """Classify duration into short/medium/long."""
        if duration <= self.SHORT_MAX:
            return "short"
        elif duration <= self.MEDIUM_MAX:
            return "medium"
        return "long"
    
    def _get_adjacent_zone(self, zone: str, progress: str) -> Optional[str]:
        """Get adjacent zone based on progress direction."""
        zone_order = [
            Zone.OWN_HALF.value,
            Zone.MID_FIELD.value,
            Zone.FINAL_THIRD.value,
            Zone.BOX_EDGE.value
        ]
        
        try:
            idx = zone_order.index(zone)
        except ValueError:
            return None
        
        if progress in ["forward", "deep_forward"] and idx < len(zone_order) - 1:
            return zone_order[idx + 1]
        elif progress == "backward" and idx > 0:
            return zone_order[idx - 1]
        
        return None


class CommentaryEditor:
    """
    LLM-based commentary editor that:
    1. Selects the best template from candidates
    2. Fills slots ({team}, {player}, {pass_count}, etc.)
    3. Ensures flow with previous commentary
    4. Does NOT invent new information
    
    Uses the LLM as an EDITOR, not a writer.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # api_key parametresi geriye donuk uyumluluk icin tutuldu.
        _ = api_key
        self.client = setup(progress_callback=lambda msg: print(f"🤖 {msg}"))
        self.model = os.getenv("MODEL_ANALYSIS", MODEL_ANALYSIS)
        self.use_mock = self.client is None

        if self.use_mock:
            print("⚠️  Gemini baglantisi kurulamadigi icin mock edit moda gecildi.")
        else:
            print(f"✅ Commentary LLM modeli: {self.model}")

    def _mock_edit(self, candidates: List[TemplateCandidate], slot_values: Dict) -> Dict:
        """LLM kullanilamazsa en guvenli fallback."""
        if not candidates:
            return {"text": "Mac devam ediyor.", "tone": "neutral", "id": ""}

        selected = random.choice(candidates)
        result = selected.text
        for slot, value in slot_values.items():
            result = result.replace(f"{{{slot}}}", str(value))

        return {"text": result, "tone": selected.tone, "id": selected.id}

    def _generate_json_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self.use_mock:
            raise RuntimeError("Gemini client hazir degil.")

        return generate_text(
            client=self.client,
            model=self.model,
            system_instruction=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
    
    def edit_commentary(self, 
                       segment: PossessionSegment,
                       candidates: List[TemplateCandidate],
                       previous_commentary: Optional[str] = None,
                       slot_values: Optional[Dict] = None,
                       used_ids: Optional[set] = None,
                       next_commentary: Optional[str] = None) -> Dict:
        """
        Edit/select commentary from candidates.
        
        Args:
            segment: PossessionSegment with context
            candidates: List of template candidates
            previous_commentary: Previous segment's commentary (for flow)
            slot_values: Values to fill slots {team}, {player}, etc.
            used_ids: Set of already used template IDs to exclude
            next_commentary: Next segment's commentary (for team name decision)
        
        Returns:
            Dict with "text", "tone", and "id" keys
        """
        if not candidates:
            return {"text": "Maç devam ediyor.", "tone": "neutral", "id": ""}
        
        # Filter out already used templates
        if used_ids:
            available = [c for c in candidates if c.id not in used_ids]
            if available:
                candidates = available
        
        slot_values = slot_values or {}
        slot_values.setdefault("team", segment.team_name)
        slot_values.setdefault("pass_count", str(segment.pass_count))
        
        return self._llm_edit(segment, candidates, previous_commentary, slot_values, next_commentary)
    
    def _llm_edit(self, segment: PossessionSegment,
                  candidates: List[TemplateCandidate],
                  previous_commentary: Optional[str],
                  slot_values: Dict,
                  next_commentary: Optional[str] = None) -> Dict:
        """Use LLM to select best template and fill slots."""
        if self.use_mock:
            return self._mock_edit(candidates, slot_values)
        
        # Build candidate list for prompt with IDs
        candidate_texts = []
        id_to_candidate = {}
        for i, c in enumerate(candidates, 1):
            candidate_texts.append(f"{i}. \"{c.text}\" (süre: {c.duration:.1f}s, ton: {c.tone})")
            id_to_candidate[i] = c
        
        # Topa sahip olan takımı belirle (segment.team_name zaten doğru takımı içeriyor)
        current_team = slot_values.get('team', segment.team_name)
        
        # Önceki ve sonraki yorumlarda takım adı geçiyor mu kontrol et
        prev_has_team = previous_commentary and (current_team in previous_commentary or "Takım" in previous_commentary)
        next_has_team = next_commentary and (current_team in next_commentary or "Takım" in next_commentary)
        
        # Bağlam bilgisi oluştur
        context_info = []
        if previous_commentary:
            context_info.append(f"ÖNCEKİ YORUM: {previous_commentary}")
        else:
            context_info.append("ÖNCEKİ YORUM: (yok - bu ilk yorum)")
        
        if next_commentary:
            context_info.append(f"SONRAKİ YORUM: {next_commentary}")
        else:
            context_info.append("SONRAKİ YORUM: (yok - bu son yorum)")
        
        # Takım adı durumu
        team_status = []
        if prev_has_team:
            team_status.append("önceki yorumda takım adı VAR")
        else:
            team_status.append("önceki yorumda takım adı YOK")
        
        if next_has_team:
            team_status.append("sonraki yorumda takım adı VAR")
        else:
            team_status.append("sonraki yorumda takım adı YOK")
        
        # Gol/şut için özel uyarı oluştur
        outcome_warning = ""
        if segment.outcome == "goal":
            outcome_warning = "\n🚨 ÖNEMLİ: Bu bir GOL anı! Mutlaka 'GOL', 'GOOOL' veya benzeri bir gol ifadesi içeren cümle SEÇ!"
        elif segment.outcome == "shot":
            outcome_warning = "\n🚨 ÖNEMLİ: Bu bir ŞUT anı! Mutlaka 'şut', 'kaleyi yokluyor' veya benzeri bir şut ifadesi içeren cümle SEÇ!"
        
        # Pas sayısına göre uyarı oluştur
        pass_count_warning = ""
        if segment.pass_count < 3:
            pass_count_warning = f"\n⚠️ DİKKAT: Sadece {segment.pass_count} pas yapıldı! 'Paslaşmalar', 'seri pas', 'üçgenler', 'dar alanda', 'pas trafiği' gibi ÇOKLU PAS ima eden cümleler SEÇME! Bu tür cümleler en az 3+ pas yapıldığında mantıklı olur."
        
        prompt = f"""Sen bir futbol spikeri editörüsün. Aşağıdaki aday cümlelerden EN UYGUN olanı seç.

SEGMENT BİLGİSİ:
- Topa sahip takım: {current_team}
- Niyet: {segment.intent} (ne tür bir pas/hareket: build_up=oyun kurma, counter=kontra, probe=açık arama)
- Bölge: {segment.zone} (own_half=kendi yarı, mid_field=orta, final_third=son bölge, box_edge=ceza sahası)
- Sonuç: {segment.outcome}
- Pas sayısı: {segment.pass_count}
- Süre: {segment.duration:.1f}s{outcome_warning}{pass_count_warning}

ADAY CÜMLELER:
{chr(10).join(candidate_texts)}

{chr(10).join(context_info)}
📌 Takım adı durumu: {', '.join(team_status)}

KURALLAR:
1. Aday cümlelerden birini SEÇ
2. Cümlede {{team}} veya {{pass_count}} varsa doldur:
   - {{team}} → {current_team}
   - {{pass_count}} → {slot_values.get('pass_count', '?')}
3. Takım adı ekleme: Hem önceki hem sonraki yorumda takım adı yoksa ve mantıklıysa "{current_team}" ekleyebilirsin. Ama zorunlu değil - doğal geliyorsa ekle, gerekmiyorsa ekleme. Önceki veya sonraki yorumda zaten takım adı varsa EKLEME.
4. Önceki ve sonraki yorumlarla çakışma veya tekrar olmasın
5. ⚠️ Sonuç 'goal' ise MUTLAKA gol ifadesi içeren cümle seç! Sonuç 'shot' ise MUTLAKA şut ifadesi içeren cümle seç!
6. ⚠️ Pas sayısı 3'ten azsa "paslaşmalar", "seri pas", "üçgenler", "dar alanda", "pas trafiği" gibi çoklu pas ifadeleri içeren cümle SEÇME!
7. JSON formatında cevap ver: {{"secim": <numara>, "cumle": "<cümle>"}}

CEVAP:"""

        try:
            raw_result = self._generate_json_reply(
                system_prompt="Sen bir futbol spikeri editorusun. Sadece JSON formatinda cevap ver.",
                user_prompt=prompt,
                temperature=0.5,
                max_tokens=100,
            )
            
            # Try to parse JSON response
            import json as json_module
            try:
                # Clean markdown code blocks if present
                if raw_result.startswith("```"):
                    raw_result = raw_result.split("```")[1]
                    if raw_result.startswith("json"):
                        raw_result = raw_result[4:]
                parsed = json_module.loads(raw_result)
                selected_num = parsed.get("secim", 1)
                result = parsed.get("cumle", "")
            except:
                # Fallback: just use the text directly
                selected_num = 1
                result = raw_result.strip('"\'')
            
            # Get the selected candidate
            selected_candidate = id_to_candidate.get(selected_num, candidates[0])
            
            # Ensure slots are filled (in case LLM missed some)
            if not result:
                result = selected_candidate.text
            for slot, value in slot_values.items():
                result = result.replace(f"{{{slot}}}", str(value))
            
            return {"text": result, "tone": selected_candidate.tone, "id": selected_candidate.id}
            
        except Exception as e:
            print(f"⚠️  LLM edit failed: {e}")
            return self._mock_edit(candidates, slot_values)


class TemplateCommentaryGenerator:
    """
    Main class that combines all components for template-based commentary.
    
    Pipeline:
    1. PossessionSegmenter: Raw events → Possession segments
    2. TemplateSelector: Segments → Candidate templates (rule-based)
    3. CommentaryEditor: Candidates → Final commentary (LLM-assisted)
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        self.bank = TemplateBank(templates_path)
        self.selector = TemplateSelector(self.bank)
        self.editor = CommentaryEditor()
        self.previous_commentary = None
        self.used_template_ids: set = set()  # Track used template IDs to prevent duplicates
        # Calculate average template duration from templates.json
        self.avg_template_duration = self.bank.get_average_template_duration()
    
    def reset(self):
        """Reset state for new video."""
        self.previous_commentary = None
        self.used_template_ids.clear()
    
    def _get_gap_commentary(self, max_duration: float, used_ids: set = None) -> Optional[Dict]:
        """
        Boşluk için gerçek bir yorum template'i seçer.
        Filler yerine önce build_up/transition yorumu kullanmayı dener.
        
        Args:
            max_duration: Maksimum yorum süresi
            used_ids: Kullanılmış template ID'leri
            
        Returns:
            Template dict veya None
        """
        # Duration range belirle
        if max_duration <= 2.0:
            duration_range = "short"
        elif max_duration <= 3.5:
            duration_range = "medium"
        else:
            duration_range = "long"
        
        # Öncelik sırası: build_up mid_field, transition, build_up own_half
        search_order = [
            ("build_up", "mid_field", "continues"),
            ("transition", "_default", "continues"),
            ("build_up", "own_half", "continues"),
        ]
        
        for intent, zone, outcome in search_order:
            templates = self.bank.get_templates(intent, zone, outcome, duration_range)
            
            if templates:
                # Kullanılmamış olanları filtrele
                if used_ids:
                    available = [t for t in templates if t.get("id", "") not in used_ids]
                    if available:
                        template = random.choice(available)
                    else:
                        continue  # Tümü kullanılmış, sonraki kategoriye geç
                else:
                    template = random.choice(templates)
                
                # Süre kontrolü
                if template.get("duration", 3.0) <= max_duration:
                    return {
                        "text": template.get("text", ""),
                        "duration": template.get("duration", 2.0),
                        "tone": template.get("tone", "neutral"),
                        "id": template.get("id", ""),
                        "intent": intent,
                        "zone": zone,
                        "outcome": outcome
                    }
        
        return None
    
    def _fill_gaps_with_fillers(self, commentaries: List[Dict], video_duration: float,
                                 max_silence: float, min_gap: float) -> List[Dict]:
        """
        3 saniyeden fazla boşluklara yorum ekler.
        
        ÖNCELİK SIRASI:
        1. Önce gerçek template yorumu (build_up, transition) eklemeye çalışır
        2. Template bulunamazsa filler ekler
        
        KURALLAR:
        - İlk yorum ve şut yorumlarının zamanları değiştirilmez
        - Şut/gol yorumunun hemen öncesine ekleme yapılmaz (0.5s buffer)
        
        Args:
            commentaries: Mevcut yorumlar (kronolojik sıralı)
            video_duration: Video süresi
            max_silence: Maksimum izin verilen boşluk (3.0s)
            min_gap: Minimum boşluk (0.5s)
            
        Returns:
            Doldurulmuş yorum listesi
        """
        if len(commentaries) < 1:
            return commentaries
        
        result = []
        total_added = 0
        used_template_ids = set()
        used_filler_ids = set()
        
        print(f"\n🔍 Boşluk doldurma kontrolü (max boşluk: {max_silence}s, video: {video_duration:.2f}s)")
        
        for i, commentary in enumerate(commentaries):
            result.append(commentary)
            
            # Son yorum - araya ekleme yapmıyoruz ama video sonu kontrolü ayrıca yapılacak
            if i >= len(commentaries) - 1:
                continue
            
            next_commentary = commentaries[i + 1]
            gap_start = commentary["end_time"]
            gap_end = next_commentary["start_time"]
            gap_duration = gap_end - gap_start
            
            # Sonraki yorum şut/gol ise, hemen öncesine ekleme yapma
            next_outcome = next_commentary.get("segment_info", {}).get("outcome", "")
            is_next_shot = next_outcome in ["shot", "goal", "shot_saved", "shot_wide", "shot_post"]
            
            # 3 saniyeden fazla boşluk var mı?
            if gap_duration > max_silence:
                print(f"   ⚠️ Boşluk tespit edildi: {gap_start:.2f}s - {gap_end:.2f}s ({gap_duration:.2f}s)")
                
                gap_items_added = 0
                
                # Şut öncesi buffer
                if is_next_shot:
                    available_end = gap_end - 0.5
                else:
                    available_end = gap_end - min_gap
                
                # Boşluğu doldur
                current_time = gap_start + min_gap
                while current_time + 1.0 <= available_end and (available_end - current_time) > max_silence / 2:
                    remaining_gap = available_end - current_time
                    max_item_duration = min(remaining_gap - min_gap, 4.5)  # Max 4.5s yorum
                    
                    if max_item_duration < 0.8:
                        break
                    
                    # ÖNCELİK 1: Gerçek template yorumu dene
                    gap_commentary = self._get_gap_commentary(max_item_duration, used_template_ids)
                    
                    if gap_commentary:
                        # Gerçek yorum bulundu
                        item_duration = gap_commentary["duration"]
                        item_start = current_time
                        item_end = item_start + item_duration
                        
                        if gap_commentary.get("id"):
                            used_template_ids.add(gap_commentary["id"])
                        
                        new_commentary = {
                            "text": gap_commentary["text"],
                            "tone": gap_commentary["tone"],
                            "start_time": round(item_start, 2),
                            "end_time": round(item_end, 2),
                            "duration": round(item_duration, 2),
                            "event_type": gap_commentary.get("intent", "build_up"),
                            "event_frame": 0,
                            "segment_info": {
                                "intent": gap_commentary.get("intent", "build_up"),
                                "zone": gap_commentary.get("zone", "mid_field"),
                                "outcome": gap_commentary.get("outcome", "continues"),
                                "tempo": "medium",
                                "pass_count": 0,
                                "is_dangerous": False,
                                "is_highlight": False
                            },
                            "candidates_count": 1,
                            "is_gap_fill": True  # Boşluk dolgusu olarak işaretle
                        }
                        
                        result.append(new_commentary)
                        total_added += 1
                        gap_items_added += 1
                        print(f"   ✅ Yorum eklendi: {item_start:.2f}s - {item_end:.2f}s \"{gap_commentary['text'][:40]}...\"")
                        
                        current_time = item_end + min_gap
                    else:
                        # ÖNCELİK 2: Filler kullan
                        filler = self.bank.get_filler(max_item_duration, used_filler_ids)
                        
                        if not filler:
                            break
                        
                        filler_duration = filler["duration"]
                        filler_start = current_time
                        filler_end = filler_start + filler_duration
                        
                        if filler.get("id"):
                            used_filler_ids.add(filler["id"])
                        
                        filler_commentary = {
                            "text": filler["text"],
                            "tone": filler["tone"],
                            "start_time": round(filler_start, 2),
                            "end_time": round(filler_end, 2),
                            "duration": round(filler_duration, 2),
                            "event_type": "filler",
                            "event_frame": 0,
                            "segment_info": {
                                "intent": "filler",
                                "zone": "mid_field",
                                "outcome": "continues",
                                "tempo": "medium",
                                "pass_count": 0,
                                "is_dangerous": False,
                                "is_highlight": False
                            },
                            "candidates_count": 1,
                            "is_filler": True
                        }
                        
                        result.append(filler_commentary)
                        total_added += 1
                        gap_items_added += 1
                        print(f"   ✅ Filler eklendi: {filler_start:.2f}s - {filler_end:.2f}s \"{filler['text']}\"")
                        
                        current_time = filler_end + min_gap
                    
                    # Bir boşluğa maksimum 5 öğe ekle
                    if gap_items_added >= 5:
                        break
        
        # ========== SON YORUMDAN VİDEO SONUNA KADAR BOŞLUK KONTROLÜ ==========
        if result:
            last_commentary = result[-1]
            last_end_time = last_commentary["end_time"]
            gap_to_end = video_duration - last_end_time
            
            # Son yorumdan video sonuna kadar boşluk var mı?
            if gap_to_end > max_silence:
                print(f"   ⚠️ Video sonu boşluğu tespit edildi: {last_end_time:.2f}s - {video_duration:.2f}s ({gap_to_end:.2f}s)")
                
                gap_items_added = 0
                available_end = video_duration - 0.5  # Video sonundan 0.5s önce bitir
                current_time = last_end_time + min_gap
                
                while current_time + 1.0 <= available_end and (available_end - current_time) > max_silence / 2:
                    remaining_gap = available_end - current_time
                    max_item_duration = min(remaining_gap - min_gap, 4.5)
                    
                    if max_item_duration < 0.8:
                        break
                    
                    # ÖNCELİK 1: Gerçek template yorumu dene
                    gap_commentary = self._get_gap_commentary(max_item_duration, used_template_ids)
                    
                    if gap_commentary:
                        item_duration = gap_commentary["duration"]
                        item_start = current_time
                        item_end = item_start + item_duration
                        
                        if gap_commentary.get("id"):
                            used_template_ids.add(gap_commentary["id"])
                        
                        new_commentary = {
                            "text": gap_commentary["text"],
                            "tone": gap_commentary["tone"],
                            "start_time": round(item_start, 2),
                            "end_time": round(item_end, 2),
                            "duration": round(item_duration, 2),
                            "event_type": gap_commentary.get("intent", "build_up"),
                            "event_frame": 0,
                            "segment_info": {
                                "intent": gap_commentary.get("intent", "build_up"),
                                "zone": gap_commentary.get("zone", "mid_field"),
                                "outcome": gap_commentary.get("outcome", "continues"),
                                "tempo": "medium",
                                "pass_count": 0,
                                "is_dangerous": False,
                                "is_highlight": False
                            },
                            "candidates_count": 1,
                            "is_gap_fill": True
                        }
                        
                        result.append(new_commentary)
                        total_added += 1
                        gap_items_added += 1
                        print(f"   ✅ Video sonu yorumu eklendi: {item_start:.2f}s - {item_end:.2f}s \"{gap_commentary['text'][:40]}...\"")
                        
                        current_time = item_end + min_gap
                    else:
                        # ÖNCELİK 2: Filler kullan
                        filler = self.bank.get_filler(max_item_duration, used_filler_ids)
                        
                        if not filler:
                            break
                        
                        filler_duration = filler["duration"]
                        filler_start = current_time
                        filler_end = filler_start + filler_duration
                        
                        if filler.get("id"):
                            used_filler_ids.add(filler["id"])
                        
                        filler_commentary = {
                            "text": filler["text"],
                            "tone": filler["tone"],
                            "start_time": round(filler_start, 2),
                            "end_time": round(filler_end, 2),
                            "duration": round(filler_duration, 2),
                            "event_type": "filler",
                            "event_frame": 0,
                            "segment_info": {
                                "intent": "filler",
                                "zone": "mid_field",
                                "outcome": "continues",
                                "tempo": "medium",
                                "pass_count": 0,
                                "is_dangerous": False,
                                "is_highlight": False
                            },
                            "candidates_count": 1,
                            "is_filler": True
                        }
                        
                        result.append(filler_commentary)
                        total_added += 1
                        gap_items_added += 1
                        print(f"   ✅ Video sonu filler eklendi: {filler_start:.2f}s - {filler_end:.2f}s \"{filler['text']}\"")
                        
                        current_time = filler_end + min_gap
                    
                    if gap_items_added >= 5:
                        break
        
        # Yeniden kronolojik sırala
        result.sort(key=lambda x: x["start_time"])
        
        if total_added > 0:
            print(f"   📝 Toplam {total_added} yorum/filler eklendi")
        else:
            print(f"   ✅ Boşluk doldurma gerekmiyor")
        
        return result

    def _postprocess_team_names(self, commentaries: List[Dict], context: Dict) -> List[Dict]:
        """
        Ardışık yorumlarda tekrarlanan takım adlarını yönetir.
        
        Mantık:
        - Önceki yorumda takım adı varsa ve mevcut yorumda da varsa
        - Mevcut yorum için TAKİM ADI İÇERMEYEN alternatif template seç
        - Bu şekilde yorum yarım kalmaz, tamamen yeni bir yorum kullanılır
        """
        if not commentaries or len(commentaries) < 2:
            return commentaries
        
        team_left = context.get('team_left', 'Manchester United')
        team_right = context.get('team_right', 'Bournemouth')
        team_names = [team_left, team_right]
        
        # Genel takım tanımlayıcıları da kontrol et
        team_indicators = team_names + ['Takım', 'takım', '{team}']
        
        def has_team_name(text: str) -> bool:
            """Metinde takım adı var mı kontrol et."""
            for indicator in team_indicators:
                if indicator in text:
                    return True
            return False
        
        def get_alternative_template_without_team(segment_info: Dict) -> Optional[str]:
            """
            Segment bilgilerine göre takım adı içermeyen alternatif template bul.
            ŞUT/GOL kategorileri için bu fonksiyon çağrılmamalı!
            """
            intent = segment_info.get('intent', 'build_up')
            zone = segment_info.get('zone', 'mid_field')
            outcome = segment_info.get('outcome', 'continues')
            
            # Şut/gol için alternatif arama - bu durumda çağrılmamalıydı ama güvenlik için
            if outcome in ['shot', 'goal', 'shot_saved', 'shot_wide', 'shot_post']:
                return None
            
            # Template bank'tan takım adı içermeyen template'leri ara
            templates_without_team = []
            
            # Intent -> zone -> outcome -> duration_range yapısında ara
            intent_data = self.bank.templates.get(intent, {})
            zone_data = intent_data.get(zone, intent_data.get("_default", {}))
            
            # Şut/gol değilse, outcome'u direkt kullan, fallback olarak continues
            # Ama şut/gol outcome varsa fallback yapma!
            if outcome in zone_data:
                outcome_data = zone_data.get(outcome, {})
            else:
                outcome_data = zone_data.get("continues", {})
            
            # Tüm duration range'leri kontrol et
            for duration_range in ["short", "medium", "long"]:
                templates = outcome_data.get(duration_range, [])
                for t in templates:
                    text = t.get('text', '')
                    # Takım adı slot'u veya sabit takım adı içermiyorsa ekle
                    if '{team}' not in text and not has_team_name(text):
                        # Daha önce kullanılmamış olmalı
                        if t.get('id') and t.get('id') not in self.used_template_ids:
                            templates_without_team.append(t)
            
            if templates_without_team:
                # Rastgele bir template seç
                import random
                selected = random.choice(templates_without_team)
                # Kullanıldı olarak işaretle
                if selected.get('id'):
                    self.used_template_ids.add(selected.get('id'))
                return selected.get('text')
            
            return None
        
        # İlk yorum hariç (ilk yorumda takım adı olabilir), diğerlerini kontrol et
        # ŞUT/GOL YORUMLARI HARİÇ - bunlar değiştirilmemeli!
        for i in range(1, len(commentaries)):
            prev_text = commentaries[i - 1].get('text', '')
            curr_text = commentaries[i].get('text', '')
            
            # Şut/gol yorumlarını atla - bunlar değiştirilmemeli!
            curr_outcome = commentaries[i].get('segment_info', {}).get('outcome', '')
            if curr_outcome in ['shot', 'goal', 'shot_saved', 'shot_wide', 'shot_post']:
                continue
            
            # Önceki yorumda takım adı varsa ve mevcut yorumda da varsa
            if has_team_name(prev_text) and has_team_name(curr_text):
                # Segment bilgilerini al
                segment_info = commentaries[i].get('segment_info', {})
                
                # Takım adı içermeyen alternatif template bul
                alt_text = get_alternative_template_without_team(segment_info)
                
                if alt_text:
                    print(f"   🔄 Alternatif template seçildi: '{curr_text[:40]}...' -> '{alt_text[:40]}...'")
                    commentaries[i]['text'] = alt_text
                else:
                    # Alternatif bulunamadıysa, takım adını kaldır (fallback)
                    for team_name in team_names:
                        if curr_text.startswith(team_name + " "):
                            remaining = curr_text[len(team_name):].strip()
                            if remaining:
                                commentaries[i]['text'] = remaining[0].upper() + remaining[1:]
                            break
                        # "'da", "'de" gibi eklerle
                        for suffix in ["'ın ", "'in ", "'un ", "'ün ", "'da ", "'de "]:
                            pattern = team_name + suffix
                            if curr_text.startswith(pattern):
                                remaining = curr_text[len(pattern):].strip()
                                if remaining:
                                    commentaries[i]['text'] = remaining[0].upper() + remaining[1:]
                                break
        
        return commentaries
    
    def generate_for_segment(self, segment: PossessionSegment,
                            slot_values: Optional[Dict] = None) -> Dict:
        """
        Generate commentary for a single possession segment.
        
        Returns:
            {
                "text": "Final commentary",
                "tone": "excited/neutral/tense/...",
                "start_time": float,
                "end_time": float,
                "duration": float,
                "segment_info": {...}
            }
        """
        # Select candidate templates
        candidates = self.selector.select_candidates(segment)
        
        # Edit/select final commentary (now returns dict with text and tone)
        result = self.editor.edit_commentary(
            segment=segment,
            candidates=candidates,
            previous_commentary=self.previous_commentary,
            slot_values=slot_values
        )
        
        self.previous_commentary = result["text"]
        
        return {
            "text": result["text"],
            "tone": result["tone"],  # ← Tone bilgisi eklendi!
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "duration": segment.duration,
            "event_type": segment.intent,
            "event_frame": segment.start_frame,
            "segment_info": {
                "intent": segment.intent,
                "zone": segment.zone,
                "outcome": segment.outcome,
                "tempo": segment.tempo,
                "pass_count": segment.pass_count,
                "is_dangerous": segment.is_dangerous,
                "is_highlight": segment.is_highlight
            },
            "candidates_count": len(candidates)
        }
    
    def generate_for_segments(self, segments: List[PossessionSegment],
                             context: Optional[Dict] = None) -> List[Dict]:
        """
        Generate commentary for a list of possession segments.
        
        Her segment için sureye sigacak birden fazla yorum uretir.
        LLM sureye gore template secer ve aralarinda dogal sessizlik birakir.
        
        Args:
            segments: List of PossessionSegment objects
            context: Optional context with team names, etc.
        
        Returns:
            List of commentary dicts
        """
        self.previous_commentary = None  # Reset for new video
        commentaries = []
        
        context = context or {}
        team_left = context.get("team_left", "Manchester United")
        team_right = context.get("team_right", "Bournemouth")
        
        for segment in segments:
            slot_values = {
                "team": segment.team_name,
                "pass_count": str(segment.pass_count)
            }
            
            # Segment için multi-commentary üret (süreye sığacak şekilde)
            segment_commentaries = self._generate_multi_commentary(segment, slot_values)
            commentaries.extend(segment_commentaries)
        
        return commentaries
    
    def _generate_multi_commentary(self, segment: PossessionSegment, 
                                   slot_values: Dict) -> List[Dict]:
        """
        Segment süresi boyunca birden fazla yorum üret.
        
        Mantık:
        - Kısa segmentler (< 2s): 1 yorum
        - Orta segmentler (2-4s): 1-2 yorum
        - Uzun segmentler (> 4s): 2-3 yorum
        - Şut/gol olan segmentlerde şut yorumu ekle
        """
        duration = segment.duration
        has_shot = segment.outcome in ["shot", "goal"]
        
        # Kaç yorum üretilecek?
        if duration < 2.0:
            target_count = 1
        elif duration < 4.0:
            target_count = 2 if has_shot or segment.pass_count >= 2 else 1
        else:
            target_count = 3 if has_shot else 2
        
        # Template adaylarını topla
        candidates = self.selector.select_candidates(segment)
        
        # Şut template'leri de ekle
        if has_shot:
            outcome_type = "goal" if segment.outcome == "goal" else "shot_saved"
            shot_templates = self.bank.get_outcome_templates(outcome_type, "short")
            for t in shot_templates[:2]:
                candidates.append(TemplateCandidate(
                    text=t.get("text", "Şut!"),
                    duration=t.get("duration", 1.0),
                    tone=t.get("tone", "excited"),
                    intent=segment.intent,
                    zone=segment.zone,
                    outcome=segment.outcome,
                    duration_range="short"
                ))
        
        if not candidates:
            return [self.generate_for_segment(segment, slot_values)]
        
        # LLM veya mock ile birden fazla yorum sec
        if self.editor.use_mock:
            return self._mock_multi_select(segment, candidates, slot_values, target_count)
        else:
            return self._llm_multi_select(segment, candidates, slot_values, target_count)
    
    def _mock_multi_select(self, segment: PossessionSegment,
                           candidates: List[TemplateCandidate],
                           slot_values: Dict,
                           target_count: int) -> List[Dict]:
        """
        Mock mode: Basitçe ilk N template'i seç ve zamanla.
        """
        results = []
        duration = segment.duration
        
        # Kullanılacak template sayısı
        use_count = min(target_count, len(candidates))
        
        # Her yorum için zaman dilimi hesapla
        time_per_comment = duration / use_count
        
        for i in range(use_count):
            template = candidates[i]
            text = template.text
            
            # Slot'ları doldur
            for slot, value in slot_values.items():
                text = text.replace(f"{{{slot}}}", str(value))
            
            # Zamanlama
            start_time = segment.start_time + (i * time_per_comment)
            end_time = start_time + min(template.duration, time_per_comment * 0.8)  # %20 sessizlik
            
            results.append({
                "text": text,
                "tone": template.tone,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "event_type": segment.intent if i < use_count - 1 or not segment.outcome in ["shot", "goal"] else "shot",
                "event_frame": segment.start_frame + int((segment.end_frame - segment.start_frame) * i / use_count),
                "segment_info": {
                    "intent": segment.intent,
                    "zone": segment.zone,
                    "outcome": segment.outcome if i == use_count - 1 else "continues",
                    "tempo": segment.tempo,
                    "pass_count": segment.pass_count,
                    "is_dangerous": segment.is_dangerous,
                    "is_highlight": segment.is_highlight
                },
                "candidates_count": len(candidates)
            })
        
        self.previous_commentary = results[-1]["text"] if results else None
        return results
    
    def _llm_multi_select(self, segment: PossessionSegment,
                          candidates: List[TemplateCandidate],
                          slot_values: Dict,
                          target_count: int) -> List[Dict]:
        """
        LLM ile segment süresi boyunca birden fazla yorum seç.
        """
        # Build candidate list for prompt
        candidate_texts = []
        for i, c in enumerate(candidates, 1):
            candidate_texts.append(f"{i}. \"{c.text}\" (süre: {c.duration:.1f}s, ton: {c.tone})")
        
        prompt = f"""Sen bir futbol spikerisin. Aşağıdaki segment için {target_count} ADET yorum seç.

SEGMENT BİLGİSİ:
- Toplam süre: {segment.duration:.1f} saniye
- Takım: {slot_values.get('team', 'Takım')}
- Pas sayısı: {segment.pass_count}
- Bölge: {segment.zone}
- Sonuç: {segment.outcome}
- Niyet: {segment.intent}

ADAY CÜMLELER:
{chr(10).join(candidate_texts)}

SLOT DEĞERLERİ (bunları doldur):
- {{team}} = {slot_values.get('team', 'Takım')}
- {{pass_count}} = {slot_values.get('pass_count', '?')}

KURALLAR:
1. Tam {target_count} adet cümle seç (numara sırasıyla)
2. Slotları doldur
3. Her cümle arası doğal sessizlik olacak
4. Şut varsa son cümle şut yorumu olsun
5. YENİ BİLGİ EKLEME!

CEVAP FORMATI (JSON):
[
  {{"text": "...", "tone": "..."}},
  {{"text": "...", "tone": "..."}}
]

CEVAP:"""

        try:
            result_text = self.editor._generate_json_reply(
                system_prompt="Sen bir futbol spikerisin. Sadece JSON formatinda cevap ver.",
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=300,
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                selected = json.loads(json_match.group())
            else:
                # Fallback to mock
                return self._mock_multi_select(segment, candidates, slot_values, target_count)
            
            # Build commentary dicts with timing
            results = []
            time_per_comment = segment.duration / len(selected)
            
            for i, sel in enumerate(selected):
                text = sel.get("text", "Maç devam ediyor.")
                tone = sel.get("tone", "neutral")
                
                # Ensure slots are filled
                for slot, value in slot_values.items():
                    text = text.replace(f"{{{slot}}}", str(value))
                
                start_time = segment.start_time + (i * time_per_comment)
                # Tahmini TTS süresi (kelime başına ~0.3s)
                word_count = len(text.split())
                estimated_duration = min(word_count * 0.3, time_per_comment * 0.8)
                end_time = start_time + estimated_duration
                
                results.append({
                    "text": text,
                    "tone": tone,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "event_type": segment.intent if i < len(selected) - 1 else segment.outcome,
                    "event_frame": segment.start_frame + int((segment.end_frame - segment.start_frame) * i / len(selected)),
                    "segment_info": {
                        "intent": segment.intent,
                        "zone": segment.zone,
                        "outcome": segment.outcome if i == len(selected) - 1 else "continues",
                        "tempo": segment.tempo,
                        "pass_count": segment.pass_count,
                        "is_dangerous": segment.is_dangerous,
                        "is_highlight": segment.is_highlight
                    },
                    "candidates_count": len(candidates)
                })
            
            self.previous_commentary = results[-1]["text"] if results else None
            return results
            
        except Exception as e:
            print(f"⚠️  LLM multi-select failed: {e}")
            return self._mock_multi_select(segment, candidates, slot_values, target_count)
    
    def reset(self):
        """Reset state for new video."""
        self.previous_commentary = None

    def generate_for_segments_smart(self, segments: List[PossessionSegment], 
                                     video_duration: float, 
                                     context: Optional[Dict] = None) -> List[Dict]:
        """
        Segment yorumlarını GERÇEK ZAMANLAMAYLA üretir:
        
        MANTIK:
        1. Önce şut/gol varsa onu shots JSON'dan gelen time + 0.5s'de yerleştir
        2. Sonra geriye doğru diğer segmentleri yerleştir (çakışmadan)
        3. Her yorumun video süresini aşmamasını garantile
        
        Args:
            segments: List of PossessionSegment
            video_duration: Toplam video süresi (saniye)
            context: Opsiyonel takım bilgileri, shots JSON verisi içerebilir
        
        Returns:
            List of commentary dicts
        """
        if not segments:
            return []
        
        self.previous_commentary = None
        self.used_template_ids.clear()
        context = context or {}
        
        MIN_GAP = 0.8  # Minimum boşluk
        video_duration = context.get("duration", 30.0)
        fps = context.get("fps", 25.0)
        
        # Shots JSON verisini al (test_shots.json'dan gelen - events.py'de zaten düzeltilmiş)
        shots_json = context.get("shots", [])
        
        print(f"\n🎬 Video süresi: {video_duration:.2f}s, FPS: {fps:.1f}")
        print(f"📊 Toplam {len(segments)} segment işlenecek")
        print(f"🎯 Shots JSON'dan {len(shots_json)} şut bilgisi alındı")
        
        # ========== ADIM 1: Segmentleri öncelik sırasına göre sırala ==========
        # Şut/gol en yüksek öncelik, sonra diğerleri kronolojik
        
        shot_segments = []
        other_segments = []
        
        for seg in segments:
            is_shot_or_goal = seg.outcome in ["goal", "shot", "shot_saved", "shot_wide", "shot_post"]
            if is_shot_or_goal:
                shot_segments.append(seg)
            else:
                other_segments.append(seg)
        
        print(f"   ⚽ Şut/Gol segmentleri: {len(shot_segments)}")
        print(f"   📦 Diğer segmentler: {len(other_segments)}")
        
        # ========== ADIM 2: Template seçimi ==========
        segment_data_map = {}
        
        # Shots JSON'dan şut zamanlarını hazırla (time değerinden)
        shot_times_from_json = {}
        for shot in shots_json:
            shot_time = shot.get("time", 0)  # test_shots.json'daki "time" değeri
            shot_frame = shot.get("frame_idx", 0)
            is_goal = shot.get("is_goal", False)
            goal_frame = shot.get("goal_frame")  # Topun çizgiyi geçtiği frame
            shot_times_from_json[shot_frame] = {
                "time": shot_time,
                "is_goal": is_goal,
                "goal_frame": goal_frame
            }
        
        for seg in segments:
            slot_values = {"team": seg.team_name, "pass_count": str(seg.pass_count)}
            
            is_shot_or_goal = seg.outcome in ["goal", "shot", "shot_saved", "shot_wide", "shot_post"]
            
            # Şut/gol için max_duration hesapla: video süresi - şut zamanı
            max_duration_for_shot = None
            shot_time_for_seg = None
            
            if is_shot_or_goal:
                # ÖNCELİK 1: Shots JSON'dan şut zamanını al (test_shots.json'daki "time" değeri)
                if shots_json:
                    # En yakın şut frame'ini bul (segment'in start/end aralığında)
                    seg_start = seg.start_frame
                    seg_end = seg.end_frame
                    best_match = None
                    best_distance = float('inf')
                    for shot_frame, shot_data in shot_times_from_json.items():
                        # Şut frame'i segment aralığında mı?
                        if seg_start <= shot_frame <= seg_end:
                            distance = 0  # Tam eşleşme
                            best_distance = distance
                            best_match = shot_data
                            break
                        else:
                            # En yakın olanı bul
                            distance = min(abs(shot_frame - seg_start), abs(shot_frame - seg_end))
                            if distance < best_distance:
                                best_distance = distance
                                best_match = shot_data
                    
                    if best_match and best_distance < 100:  # 100 frame tolerans
                        shot_time = best_match["time"]
                        shot_frame = best_match.get("frame_idx", int(shot_time * fps))
                        
                        # Şut zamanı doğrudan kullan (ortalama hesaplama kaldırıldı)
                        shot_time_for_seg = shot_time
                        print(f"   🎯 Şut zamanı: {shot_time_for_seg:.2f}s")
                
                # ÖNCELİK 2: Fallback - segment frame'lerinden hesapla
                if shot_time_for_seg is None:
                    seg_shot_time = seg.start_frame / fps
                    shot_time_for_seg = seg_shot_time
                    print(f"   🎯 Şut zamanı (segment'ten): {shot_time_for_seg:.2f}s")
                
                # Kalan süre = video süresi - şut zamanı (0.2s güvenlik payı)
                max_duration_for_shot = video_duration - shot_time_for_seg - 0.2
                if max_duration_for_shot < 1.0:
                    max_duration_for_shot = 1.0  # Minimum 1 saniye
                print(f"   ⏱️ Kalan süre: {max_duration_for_shot:.2f}s")
            
            # Şut zamanını segment'e kaydet (yerleştirmede kullanılacak)
            if shot_time_for_seg is not None:
                seg._shot_time_calculated = shot_time_for_seg
            
            # Template seçimi - max_duration varsa ona göre akıllı seçim yapılır
            candidates = self.selector.select_candidates(
                seg, 
                used_ids=self.used_template_ids,
                prefer_no_slots=False,
                prefer_short=False,  # Artık prefer_short kullanılmıyor
                max_duration=max_duration_for_shot  # Kalan süreye göre akıllı seçim
            )
            
            # ⚠️ GOL/ŞUT için SADECE gol/şut template'lerini tut
            if is_shot_or_goal and candidates:
                goal_keywords = ["gol", "GOL", "GOOL", "ağları", "köşe", "vuruş", "kurtarış", "kaleci", "şut", "ŞUT", "Şut"]
                print(f"   🔍 DEBUG: {len(candidates)} candidate before filter")
                for c in candidates[:3]:
                    print(f"      - {c.text[:40]}...")
                goal_candidates = [c for c in candidates if any(kw in c.text for kw in goal_keywords)]
                print(f"   🔍 DEBUG: {len(goal_candidates)} goal_candidates after filter")
                if goal_candidates:
                    candidates = goal_candidates
                    for c in candidates[:3]:
                        print(f"      ✓ {c.text[:40]}...")
                else:
                    print(f"   ⚠️ DEBUG: No goal keywords found, keeping original candidates")
            
            if candidates:
                # Şut/Gol için kalan süreye sığan EN UZUN template'i seç
                if is_shot_or_goal and max_duration_for_shot:
                    # Süreye sığanları filtrele (0.3s tolerans)
                    fitting_candidates = [c for c in candidates if c.duration <= max_duration_for_shot + 0.3]
                    if fitting_candidates:
                        # Sığanlar arasından EN UZUNUNU seç (daha zengin yorum)
                        best_candidate = max(fitting_candidates, key=lambda c: c.duration)
                        print(f"   ✅ Şut/Gol: Kalan {max_duration_for_shot:.1f}s → {best_candidate.duration:.1f}s template seçildi")
                        print(f"      → \"{best_candidate.text[:50]}...\"")
                    else:
                        # Hiçbiri sığmıyor - en kısasını seç
                        best_candidate = min(candidates, key=lambda c: c.duration)
                        print(f"   ⚠️ Şut/Gol: Kalan {max_duration_for_shot:.1f}s yetersiz, {best_candidate.duration:.1f}s template kullanılacak")
                else:
                    # Diğer segmentler için rastgele seç
                    best_candidate = random.choice(candidates)
                
                if best_candidate.id:
                    self.used_template_ids.add(best_candidate.id)
                
                # Basit slot doldurma
                preliminary_text = best_candidate.text
                if "{team}" in preliminary_text:
                    preliminary_text = preliminary_text.replace("{team}", seg.team_name)
                if "{pass_count}" in preliminary_text:
                    preliminary_text = preliminary_text.replace("{pass_count}", str(seg.pass_count))
                
                segment_data_map[id(seg)] = {
                    "segment": seg,
                    "candidates": candidates,
                    "slot_values": slot_values,
                    "text": preliminary_text,
                    "duration": best_candidate.duration,
                    "tone": best_candidate.tone,
                    "template_id": best_candidate.id
                }
            else:
                segment_data_map[id(seg)] = {
                    "segment": seg,
                    "candidates": [],
                    "slot_values": slot_values,
                    "text": "Maç devam ediyor.",
                    "duration": 2.0,
                    "tone": "neutral",
                    "template_id": ""
                }
        
        # ========== ADIM 3: ZAMANLAMA - Şut/gol önce, geriye doğru yerleştir ==========
        # 
        # Yeni mantık:
        # 1. Şut/gol segmentinin GERÇEK zamanına yorumu yerleştir
        # 2. Yorumun video süresini aşmamasını garantile
        # 3. Geriye doğru diğer yorumları yerleştir (çakışma olmadan)
        
        placed_commentaries = []
        used_time_ranges = []  # (start, end) tuples
        
        def time_range_conflicts(start: float, end: float) -> bool:
            """Check if a time range conflicts with already placed commentaries."""
            for placed_start, placed_end in used_time_ranges:
                # Çakışma var mı?
                if not (end <= placed_start - MIN_GAP or start >= placed_end + MIN_GAP):
                    return True
            return False
        
        def find_safe_start_time(desired_start: float, duration: float) -> float:
            """Find a safe start time that doesn't conflict and doesn't exceed video."""
            start = desired_start
            end = start + duration
            
            # Video süresini aşmamalı
            if end > video_duration - 0.1:
                start = video_duration - 0.1 - duration
                end = video_duration - 0.1
            
            # Negatif olamaz
            if start < 0.1:
                start = 0.1
                end = start + duration
            
            return start
        
        # 3a. ÖNCELİK 1: Şut/Gol yorumlarını yerleştir (shots JSON'dan gelen zamana göre)
        for seg in shot_segments:
            data = segment_data_map[id(seg)]
            
            # ŞUT ZAMANI: Daha önce hesaplanan _shot_time_calculated değerini kullan
            # Bu değer events.py'de düzeltilmiş şut zamanı veya gol çizgisi ortası
            if hasattr(seg, '_shot_time_calculated') and seg._shot_time_calculated is not None:
                shot_time = seg._shot_time_calculated
            else:
                # Fallback: segment frame zamanını kullan
                shot_time = seg.start_frame / fps
            
            duration = data["duration"]
            
            # Yorum hesaplanan zamanda başlasın
            start = shot_time
            end = start + duration
            
            # Video süresini aşıyorsa sadece bitiş zamanını ve süreyi kısalt
            if end > video_duration - 0.1:
                end = video_duration - 0.1
                duration = end - start
                # Minimum 1 saniye yorum olsun
                if duration < 1.0:
                    # Çok kısa kalıyorsa biraz geriye al ama şut anından önce başlamasın
                    start = max(shot_time - 0.5, end - 1.5)  # En fazla 0.5s önce başlayabilir
                    duration = end - start
            
            print(f"   ⚽ Şut yorumu: {start:.2f}s - {end:.2f}s (gerçek şut anı: {shot_time:.2f}s)")
            
            placed_commentaries.append({
                "text": data["text"],
                "tone": data["tone"],
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "duration": round(duration, 2),
                "event_type": seg.intent,
                "event_frame": seg.start_frame,
                "segment_info": {
                    "intent": seg.intent,
                    "zone": seg.zone,
                    "outcome": seg.outcome,
                    "tempo": seg.tempo,
                    "pass_count": seg.pass_count,
                    "is_dangerous": seg.is_dangerous,
                    "is_highlight": seg.is_highlight
                },
                "candidates_count": len(data["candidates"])
            })
            used_time_ranges.append((start, end))
        
        # 3b. DİĞER YORUMLARI YERLEŞTİR
        # Diğer segmentleri kronolojik sırala
        other_segments_sorted = sorted(other_segments, 
                                       key=lambda s: s.start_time if hasattr(s, 'start_time') else (s.start_frame / fps))
        
        # ========== ŞUT VARSA: Eski mantık - eşit dağıtım ==========
        if shot_segments and other_segments_sorted:
            MAX_SILENCE = 2.0  # Maksimum sessizlik süresi
            
            # Şut yorumunun başlangıç zamanını bul
            shot_start_time = min([r[0] for r in used_time_ranges])
            
            # Kullanılabilir zaman aralığı
            available_start = 0.1
            available_end = shot_start_time - MIN_GAP
            available_duration = available_end - available_start
            
            print(f"\n   📍 Şut öncesi yerleştirme: {available_start:.2f}s - {available_end:.2f}s ({available_duration:.2f}s)")
            
            # Toplam yorum süresini hesapla
            total_commentary_duration = sum([segment_data_map[id(seg)]["duration"] for seg in other_segments_sorted])
            total_silence = available_duration - total_commentary_duration
            
            if total_silence < 0:
                print(f"   ⚠️ Yorumlar sığmıyor! Toplam: {total_commentary_duration:.2f}s, Alan: {available_duration:.2f}s")
                # Kısaltarak sığdırmayı dene
                if len(other_segments_sorted) > 2:
                    other_segments_sorted = [other_segments_sorted[0], other_segments_sorted[-1]]
                    total_commentary_duration = sum([segment_data_map[id(seg)]["duration"] for seg in other_segments_sorted])
                    total_silence = available_duration - total_commentary_duration
            
            # Yorumlar arası boşlukları hesapla
            num_gaps = len(other_segments_sorted)
            if num_gaps > 0 and total_silence > 0:
                silence_per_gap = total_silence / num_gaps
                if silence_per_gap > MAX_SILENCE:
                    silence_per_gap = MAX_SILENCE
                print(f"   🔇 Yorum arası sessizlik: {silence_per_gap:.2f}s")
            else:
                silence_per_gap = MIN_GAP
            
            # Yorumları sırayla yerleştir
            current_time = available_start
            
            for i, seg in enumerate(other_segments_sorted):
                data = segment_data_map[id(seg)]
                duration = data["duration"]
                
                start = current_time
                end = start + duration
                
                # Şut yorumuyla çakışma kontrolü
                if end > shot_start_time - MIN_GAP:
                    available_for_this = shot_start_time - MIN_GAP - start
                    if available_for_this >= 1.0:
                        end = start + available_for_this
                        duration = available_for_this
                        print(f"   📝 Yorum (kısaltılmış): {start:.2f}s - {end:.2f}s")
                    else:
                        print(f"   ⚠️ Segment atlandı (şutla çakışıyor): {seg.intent}")
                        continue
                else:
                    print(f"   📝 Yorum: {start:.2f}s - {end:.2f}s")
                
                placed_commentaries.append({
                    "text": data["text"],
                    "tone": data["tone"],
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "duration": round(duration, 2),
                    "event_type": seg.intent,
                    "event_frame": seg.start_frame,
                    "segment_info": {
                        "intent": seg.intent,
                        "zone": seg.zone,
                        "outcome": seg.outcome,
                        "tempo": seg.tempo,
                        "pass_count": seg.pass_count,
                        "is_dangerous": seg.is_dangerous,
                        "is_highlight": seg.is_highlight
                    },
                    "candidates_count": len(data["candidates"])
                })
                used_time_ranges.append((start, end))
                current_time = end + silence_per_gap
        
        # ========== ŞUT YOKSA: Yeni mantık - segment zamanlarına göre yerleştir ==========
        elif other_segments_sorted:
            MIN_COMMENTARY_DURATION = 1.5
            
            print(f"\n   📍 {len(other_segments_sorted)} segment yerleştirilecek (şut yok, video: {video_duration:.2f}s)")
            
            for i, seg in enumerate(other_segments_sorted):
                data = segment_data_map[id(seg)]
                duration = data["duration"]
                
                # Segment'in kendi zamanı
                segment_time = seg.start_frame / fps
                desired_start = segment_time
                
                # İlk segment için minimum 0.1s'den başla
                if i == 0 and desired_start < 0.1:
                    desired_start = 0.1
                
                # Önceki yorumla çakışma kontrolü
                if used_time_ranges:
                    last_end = max([r[1] for r in used_time_ranges if r[1] <= desired_start + duration], default=0)
                    if desired_start < last_end + MIN_GAP:
                        desired_start = last_end + MIN_GAP
                
                start = desired_start
                end = start + duration
                
                # Video süresini aşma kontrolü
                if end > video_duration - 0.3:
                    available = video_duration - 0.3 - start
                    if available >= MIN_COMMENTARY_DURATION:
                        end = start + available
                        duration = available
                    else:
                        start = video_duration - 0.3 - duration
                        if start < 0.1:
                            start = 0.1
                        end = start + duration
                        
                        conflicts = False
                        for placed_start, placed_end in used_time_ranges:
                            if not (end <= placed_start - MIN_GAP or start >= placed_end + MIN_GAP):
                                conflicts = True
                                break
                        
                        if conflicts:
                            print(f"   ⚠️ Segment atlandı (yer yok): {seg.intent} @ {segment_time:.2f}s")
                            continue
                
                # Mevcut yerleşimlerle çakışma kontrolü
                conflicts = False
                for placed_start, placed_end in used_time_ranges:
                    if not (end <= placed_start - MIN_GAP or start >= placed_end + MIN_GAP):
                        conflicts = True
                        start = placed_end + MIN_GAP
                        end = start + duration
                        if end > video_duration - 0.3:
                            available = video_duration - 0.3 - start
                            if available >= MIN_COMMENTARY_DURATION:
                                end = start + available
                                duration = available
                                conflicts = False
                            else:
                                print(f"   ⚠️ Segment atlandı (çakışma): {seg.intent} @ {segment_time:.2f}s")
                                break
                        else:
                            conflicts = False
                        break
                
                if conflicts:
                    continue
                
                print(f"   📝 Yorum: {start:.2f}s - {end:.2f}s ({seg.intent}, segment: {segment_time:.2f}s)")
                
                placed_commentaries.append({
                    "text": data["text"],
                    "tone": data["tone"],
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "duration": round(duration, 2),
                    "event_type": seg.intent,
                    "event_frame": seg.start_frame,
                    "segment_info": {
                        "intent": seg.intent,
                        "zone": seg.zone,
                        "outcome": seg.outcome,
                        "tempo": seg.tempo,
                        "pass_count": seg.pass_count,
                        "is_dangerous": seg.is_dangerous,
                        "is_highlight": seg.is_highlight
                    },
                    "candidates_count": len(data["candidates"])
                })
                used_time_ranges.append((start, end))
        
        # Kronolojik sırala
        placed_commentaries.sort(key=lambda x: x["start_time"])
        
        print(f"\n✅ Toplam {len(placed_commentaries)} yorum yerleştirildi")
        if placed_commentaries:
            last_end = placed_commentaries[-1]["end_time"]
            print(f"   Son yorum bitiş: {last_end:.2f}s (video: {video_duration:.2f}s)")
            if last_end > video_duration:
                print(f"   ⚠️ UYARI: Son yorum video süresini aşıyor!")
            
            # Şut/gol yorumu son yorum mu kontrol et
            last_commentary = placed_commentaries[-1]
            if last_commentary.get("segment_info", {}).get("outcome") in ["shot", "goal", "shot_saved", "shot_wide", "shot_post"]:
                print(f"   ✅ Son yorum şut/gol yorumu - sonrasında ek yorum yok")
            else:
                print(f"   ⚠️ Son yorum şut/gol değil: {last_commentary.get('segment_info', {}).get('outcome')}")
        
        # ========== ADIM 4: BOŞLUK DOLDURMA - 3 saniyeden fazla boşluklara yorum/filler ekle ==========
        MAX_SILENCE = 3.0  # Maksimum sessizlik süresi
        placed_commentaries = self._fill_gaps_with_fillers(
            placed_commentaries, 
            video_duration, 
            MAX_SILENCE,
            MIN_GAP
        )
        
        # Post-processing: Ardışık takım adlarını kaldır
        placed_commentaries = self._postprocess_team_names(placed_commentaries, context)
        
        return placed_commentaries


# Convenience function for quick use
def generate_template_commentary(segments: List[Dict], context: Dict) -> List[Dict]:
    """
    Generate commentary from possession segment dicts.
    
    Args:
        segments: List of possession segment dicts
        context: {team_left, team_right, ...}
    
    Returns:
        List of commentary dicts
    """
    from .possession_segmenter import PossessionSegment
    
    # Convert dicts to PossessionSegment objects
    segment_objects = []
    for seg_dict in segments:
        segment = PossessionSegment(**seg_dict)
        segment_objects.append(segment)
    
    generator = TemplateCommentaryGenerator()
    
    return generator.generate_for_segments(segment_objects, context)


# For testing
if __name__ == "__main__":
    from .possession_segmenter import PossessionSegment
    
    # Test with sample segment
    test_segment = PossessionSegment(
        start_time=1.0,
        end_time=4.5,
        start_frame=30,
        end_frame=135,
        duration=3.5,
        team_id="R",
        team_name="Bournemouth",
        pass_count=3,
        dribble_count=1,
        total_events=4,
        intent="probe",
        tempo="medium",
        zone="final_third",
        zone_start="mid_field",
        zone_end="box_edge",
        progress="forward",
        pressure="medium",
        outcome="shot",
        avg_pass_time=0.9,
        field_x_start=55.0,
        field_x_end=88.0,
        field_progress=33.0,
        pass_sequence=["medium", "short", "dribble", "shot"],
        is_dangerous=True,
        is_highlight=True
    )
    
    generator = TemplateCommentaryGenerator()
    result = generator.generate_for_segment(test_segment)
    
    print("Test Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
