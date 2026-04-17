# 3.1.3 VAR Package (var_engine)

---

## 3.1.3.1 VARClipIngestor

| **class VARClipIngestor** |
|:--|
| Video veya görüntü dosyasını alarak doğrulamasını yapar ve LLM API üzerinden pozisyonun metin açıklamasını (description) çıkarır. Dosya boyutu (maks. 20 MB) ve MIME tipi kontrolü yapıldıktan sonra, video base64 olarak encode edilip AI modeline gönderilir. Model, pozisyonun türü, oyuncular, temas bilgisi, topun konumu ve şiddeti gibi bilgileri Türkçe metin olarak döndürür. Safety settings kontrolleri de bu sınıf tarafından yönetilir. |
| **Attributes** |
| private bytes video_bytes |
| private str mime_type |
| private str filename |
| private float file_size_mb |
| private float MAX_FILE_SIZE_MB |
| private list\<str\> SUPPORTED_MIME_TYPES |
| private GenerativeModel ai_model |
| private list\<dict\> safety_settings |
| **Methods** |
| + validate() : bool — Dosya boyutu ve MIME tipi doğrulaması yapar. |
| + ingest() : str — Ana giriş noktası: doğrula → çıkar → açıklama döndür. |
| + extract_description() : str — LLM API'ye video gönderip metin açıklama alır. |
| + get_video_part() : dict — Base64 encode edilmiş inline_data dict'i döndürür. |
| + get_file_info() : dict — Dosya bilgilerini (ad, boyut, tip) döndürür. |
| - _check_file_size() : bool — Boyut kontrolü (≤ 20 MB). |
| - _check_mime_type() : bool — MIME tipi kontrolü. |
| - _encode_base64() : str — video_bytes → Base64 string dönüşümü. |
| - _build_extraction_prompt() : str — Pozisyon analiz prompt'u oluşturur. |
| - _parse_response(response) : str — LLM yanıtını parse eder, safety kontrol yapar. |

---

## 3.1.3.2 RuleRetriever

| **class RuleRetriever** |
|:--|
| Vektör veritabanı üzerinden IFAB futbol kurallarını arar ve döndürür. İlk başlatmada PDF dosyasından (oyun-kurallari.pdf) sayfa bazlı chunk'lar çıkarılarak vektör veritabanına indexlenir. PDF bulunamazsa hardcoded IFAB kuralları (6 kural: Ofsayt, Fauller, Handball, DOGSO, Penaltı, VAR Protokolü) kullanılır. Arama sırasında sorgu metni vektör benzerliğiyle eşleşen en ilgili kurallar döndürülür. Manuel kural seçimi yapıldığında frontend seçimini kural ID'sine eşler (örn: "law12" → "law12_fauller"). |
| **Attributes** |
| private PersistentClient vector_db_client |
| private Collection rules_collection |
| private list\<dict\> IFAB_RULES |
| private str pdf_path |
| private bool is_initialized |
| private int collection_count |
| **Methods** |
| + init_rag() : dict — RAG sistemini başlatır (PDF → vektör veritabanı veya hardcoded fallback). |
| + search(query: str, n_results: int) : SearchResult — Sorguya en uygun kuralları arar. |
| + get_rule_by_id(rule_id: str) : str — Belirli kural ID'sine göre kural içeriğini döndürür. |
| + get_rule_by_selection(selection: str) : tuple — Frontend seçimini kural ID + talimat çiftine eşler. |
| + get_status() : dict — RAG durumunu (count, rules listesi) döndürür. |
| + get_collection_count() : int — Koleksiyondaki döküman sayısını döndürür. |
| - _load_from_pdf() : list\<dict\> — PDF'ten sayfa bazlı chunk'lar çıkarır. |
| - _load_hardcoded_rules() : None — Hardcoded kuralları vektör veritabanına ekler. |
| - _extract_text_from_pdf(pdf_path: str) : list\<dict\> — PDF metin çıkarma işlemi. |
| - _chunk_pages(reader: PdfReader) : list\<dict\> — Sayfa bazlı chunking. |
| - _upsert_to_collection(chunks: list) : None — Chunk'ları vektör veritabanına yazar. |

---

## 3.1.3.3 DecisionExplainer

| **class DecisionExplainer** |
|:--|
| Video, metin açıklama ve ilgili IFAB kurallarını birlikte değerlendirerek nihai VAR kararını üretir. LLM API'ye profesyonel FIFA sertifikalı VAR hakemi rolüyle prompt gönderilir. Manuel kural seçimi aktifse sadece seçilen kural kapsamında değerlendirme yapılır. Yanıt JSON formatında parse edilerek TechnicalAnalysis, RuleInterpretation, VARAssessment ve Decision alt modellerine ayrılır. Kart kararları IFAB Law 12'ye göre (Dikkatsiz → Faul, Tedbirsiz → Sarı, Aşırı Güç → Kırmızı) belirlenir. |
| **Attributes** |
| private GenerativeModel ai_model |
| private list\<dict\> safety_settings |
| private str model_name |
| **Methods** |
| + make_decision(video_bytes, mime_type, description, rag_rules, manual_selection) : AnalysisResult — Ana karar verme fonksiyonu. |
| + build_prompt(description, rag_rules, manual_selection) : str — Analiz prompt'unu oluşturur. |
| + parse_decision(response_text: str) : AnalysisResult — JSON yanıtı parse eder. |
| + get_confidence(result: AnalysisResult) : int — Güven skorunu döndürür (0-100). |
| - _build_strict_instruction(manual_selection: str) : str — Manuel seçim kısıtlama talimatı. |
| - _build_card_rules_prompt() : str — Kart kuralları prompt bloğu. |
| - _build_json_schema_prompt() : str — JSON çıktı şema tanımı. |
| - _extract_json(text: str) : dict — Yanıttan JSON bloğu çıkarır. |
| - _validate_result(parsed: dict) : bool — Zorunlu alanları doğrular. |
| - _create_fallback_result(error: str) : AnalysisResult — Hata durumunda varsayılan sonuç. |

---

| **class TechnicalAnalysis** |
|:--|
| Pozisyondaki fiziksel temasın teknik detaylarını tutar. |
| **Attributes** |
| private bool hasContact |
| private str contactType |
| private str severity |
| private str location |
| **Methods** |
| Getter and setter methods. |

---

| **class RuleInterpretation** |
|:--|
| Uygulanan IFAB kuralının yorumunu tutar. |
| **Attributes** |
| private str rule |
| private str ruleName |
| private str explanation |
| **Methods** |
| Getter and setter methods. |

---

| **class VARAssessment** |
|:--|
| VAR müdahale değerlendirmesini tutar. |
| **Attributes** |
| private bool interventionNeeded |
| private str reason |
| **Methods** |
| Getter and setter methods. |

---

| **class Decision** |
|:--|
| Nihai VAR kararını tutar. |
| **Attributes** |
| private str type |
| private str main |
| private str sub |
| private str icon |
| **Methods** |
| Getter and setter methods. |

---

| **class AnalysisResult** |
|:--|
| DecisionExplainer'ın ürettiği tüm analiz sonuçlarını bir arada tutar. |
| **Attributes** |
| private str summary |
| private TechnicalAnalysis technicalAnalysis |
| private RuleInterpretation ruleInterpretation |
| private VARAssessment varAssessment |
| private Decision decision |
| private int confidence |
| private str notes |
| **Methods** |
| Getter and setter methods. |

---

## 3.1.3.4 EvidencePackager

| **class EvidencePackager** |
|:--|
| DecisionExplainer'dan gelen ham analiz sonuçlarını frontend'e uygun HTML formatına dönüştürür ve API yanıtını paketler. TechnicalAnalysis HTML `<ul>` listesine, RuleInterpretation ve VARAssessment HTML paragraflarına çevrilir. Decision nesnesine uygun ikon (⚽🟨🟥▶️ vb.) eklenir. Severity kodları Türkçe metne çevrilir (light → "Düşük - Normal oyun teması", severe → "Yüksek - Tehlikeli müdahale"). Eksik alanlar varsayılan değerlerle doldurulur. |
| **Attributes** |
| private dict SEVERITY_MAP |
| private dict ICON_MAP |
| private dict DEFAULT_DECISION |
| **Methods** |
| + package(result: AnalysisResult, metadata: AnalysisMetadata) : FormattedResponse — Ana paketleme fonksiyonu. |
| + format_technical_analysis(ta: TechnicalAnalysis) : str — TechnicalAnalysis → HTML listesi. |
| + format_rule_interpretation(ri: RuleInterpretation) : str — RuleInterpretation → HTML paragrafları. |
| + format_var_assessment(va: VARAssessment) : str — VARAssessment → HTML paragrafları. |
| + format_decision(decision: Decision) : Decision — Decision'a ikon ekler. |
| + build_api_response(formatted, metadata) : dict — Frontend'e gönderilecek tam API yanıtı. |
| - _severity_to_text(severity: str) : str — Severity kodunu Türkçe metne çevirir. |
| - _decision_to_icon(decision_type: str) : str — Karar tipini ikona eşler. |
| - _ensure_defaults(result: dict) : dict — Eksik alanları varsayılanlarla doldurur. |

---

| **class AnalysisMetadata** |
|:--|
| Analiz süreç bilgilerini (zamanlama, kaynaklar) tutar. |
| **Attributes** |
| private str video_description |
| private list\<str\> rag_sources |
| private float step1_time |
| private float step2_time |
| private float step3_time |
| private float total_time |
| private str selected_rule |
| **Methods** |
| Getter and setter methods. |

---

| **class FormattedResponse** |
|:--|
| Frontend'e gönderilecek formatlanmış yanıtı tutar. |
| **Attributes** |
| private str summary |
| private str technicalAnalysis |
| private str ruleInterpretation |
| private str varAssessment |
| private Decision decision |
| private int confidence |
| private str notes |
| **Methods** |
| Getter and setter methods. |
