import React, { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import "./rotunda.css";

interface Project {
  id: string;
  number: string;
  category: string;
  title: string;
  tagline: string;
  port: number | null;
  url: string | null;
  hue: number;
  chroma: number;
  comingSoon?: boolean;
}

const PROJECTS: Project[] = [
  {
    id: "ai-spiker",
    number: "01",
    category: "Anlatıcı",
    title: "AI Spiker",
    tagline:
      "Maç videosunu anında spor spikeri tonunda, zamanlı bir anlatıma dönüştürür.",
    port: 5173,
    url: "http://127.0.0.1:5173",
    hue: 72,
    chroma: 0.145,
  },
  {
    id: "mac-chatbot",
    number: "02",
    category: "Diyalog",
    title: "Maç Chatbot",
    tagline:
      "Maçı konuşarak gezin; cevap gelir ve video otomatik olarak ilgili saniyeye sarar.",
    port: 5000,
    url: "http://127.0.0.1:5000",
    hue: 152,
    chroma: 0.118,
  },
  {
    id: "var-engine",
    number: "03",
    category: "Karar",
    title: "VAR Engine",
    tagline:
      "Pozisyonu teknik olarak çözümler, IFAB maddesiyle birlikte hakem kararını yazar.",
    port: 8001,
    url: "http://127.0.0.1:8001",
    hue: 28,
    chroma: 0.175,
  },
  {
    id: "simulasyon",
    number: "04",
    category: "Deneyim",
    title: "Simülasyon",
    tagline:
      "Önemli anlar Unity'de yeniden kurulur, sahaya oyuncunun gözünden girersin.",
    port: null,
    url: null,
    hue: 258,
    chroma: 0.165,
    comingSoon: true,
  },
];

const SLOT_X_VW = 26;
const SLOT_Z = 210;
const SLOT_ROT = 32;
const SLOT_SCALE = 0.83;
const BACK_Z = 560;
const BACK_SCALE = 0.58;
const BACK_LIFT = "2.6rem";

function slotOf(i: number, focus: number, n: number) {
  return (i - focus + n) % n;
}

function slotTransform(slot: number): string {
  switch (slot) {
    case 0:
      return "translate(-50%, -50%) translateZ(0) rotateY(0deg) scale(1)";
    case 1:
      return `translate(-50%, -50%) translateX(${SLOT_X_VW}vw) translateZ(-${SLOT_Z}px) rotateY(-${SLOT_ROT}deg) scale(${SLOT_SCALE})`;
    case 3:
      return `translate(-50%, -50%) translateX(-${SLOT_X_VW}vw) translateZ(-${SLOT_Z}px) rotateY(${SLOT_ROT}deg) scale(${SLOT_SCALE})`;
    default:
      // slot 2: back
      return `translate(-50%, -50%) translateY(-${BACK_LIFT}) translateZ(-${BACK_Z}px) rotateY(0deg) scale(${BACK_SCALE})`;
  }
}

function slotOpacity(slot: number): number {
  if (slot === 0) return 1;
  if (slot === 2) return 0.14;
  return 0.55;
}

function slotZIndex(slot: number): number {
  if (slot === 0) return 40;
  if (slot === 2) return 10;
  return 20;
}

function useClock() {
  const [time, setTime] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setTime(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);
  return time;
}

function readFocusFromURL(): number {
  if (typeof window === "undefined") return 0;
  const params = new URLSearchParams(window.location.search);
  const raw = params.get("focus");
  if (!raw) return 0;
  const byId = PROJECTS.findIndex((p) => p.id === raw);
  if (byId >= 0) return byId;
  const asNum = Number(raw);
  if (Number.isInteger(asNum) && asNum >= 0 && asNum < PROJECTS.length) return asNum;
  return 0;
}

export default function Rotunda() {
  const [focus, setFocus] = useState<number>(readFocusFromURL);
  const [nudge, setNudge] = useState(0);
  const focusRef = useRef(focus);
  focusRef.current = focus;
  const touchStartX = useRef<number | null>(null);
  const wheelLock = useRef(false);
  const time = useClock();

  const focused = PROJECTS[focus];

  const go = useCallback((delta: number) => {
    setFocus((prev) => (prev + delta + PROJECTS.length) % PROJECTS.length);
  }, []);

  const openFocused = useCallback(() => {
    const p = PROJECTS[focusRef.current];
    if (p.comingSoon || !p.url) {
      setNudge((n) => n + 1);
      return;
    }
    window.open(p.url, "_blank", "noopener,noreferrer");
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        go(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        go(-1);
      } else if (e.key === "Enter") {
        const tgt = e.target as HTMLElement | null;
        if (tgt?.closest("button,a,input,textarea")) return;
        e.preventDefault();
        openFocused();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, openFocused]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("focus", PROJECTS[focus].id);
    const next = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState(null, "", next);
  }, [focus]);

  const onTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX;
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    if (touchStartX.current == null) return;
    const dx = e.changedTouches[0].clientX - touchStartX.current;
    if (Math.abs(dx) > 40) go(dx < 0 ? 1 : -1);
    touchStartX.current = null;
  };

  const onWheel = (e: React.WheelEvent) => {
    if (Math.abs(e.deltaX) < 24 && Math.abs(e.deltaY) < 24) return;
    if (wheelLock.current) return;
    wheelLock.current = true;
    const primary = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
    go(primary > 0 ? 1 : -1);
    window.setTimeout(() => {
      wheelLock.current = false;
    }, 520);
  };

  const sceneStyle = useMemo<CSSProperties>(
    () =>
      ({
        "--focus-h": String(focused.hue),
        "--focus-c": String(focused.chroma),
      }) as CSSProperties,
    [focused.hue, focused.chroma]
  );

  const timeLabel = useMemo(() => {
    return time.toLocaleTimeString("tr-TR", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }, [time]);

  return (
    <div className="rotunda" style={sceneStyle} onWheel={onWheel}>
      <div className="rotunda-grain" aria-hidden="true" />
      <div className="rotunda-pitch" aria-hidden="true" />

      <header className="rotunda-header">
        <div className="rotunda-mark">
          Match Studio<small>bitirme · 2026</small>
        </div>
        <div className="rotunda-meta">
          <span>
            <em>{String(focus + 1).padStart(2, "0")}</em> / {String(PROJECTS.length).padStart(2, "0")}
          </span>
          <span>Ist · {timeLabel}</span>
        </div>
      </header>

      <div className="rotunda-stage-wrap">
        <div className="rotunda-eyebrow">Dört araç · Tek sahne</div>

        <div
          className="rotunda-scene"
          onTouchStart={onTouchStart}
          onTouchEnd={onTouchEnd}
        >
          <button
            className="rotunda-nav prev"
            aria-label="Önceki projeye dön"
            onClick={() => go(-1)}
          >
            ‹
          </button>
          <button
            className="rotunda-nav next"
            aria-label="Sonraki projeye dön"
            onClick={() => go(1)}
          >
            ›
          </button>

          <div className="rotunda-floor" aria-hidden="true" />

          {PROJECTS.map((p, i) => {
            const slot = slotOf(i, focus, PROJECTS.length);
            const isFocus = slot === 0;
            const isBack = slot === 2;
            const transform = slotTransform(slot);
            const isNudging = isFocus && p.comingSoon && nudge > 0;
            const style = {
              left: "50%",
              top: "50%",
              "--t": transform,
              opacity: slotOpacity(slot),
              zIndex: slotZIndex(slot),
              pointerEvents: isBack ? "none" : "auto",
            } as CSSProperties;

            const cls = [
              "rotunda-card",
              isFocus ? "is-focus" : isBack ? "is-back" : "is-side",
              p.comingSoon ? "is-soon" : "",
              isNudging ? "is-nudging" : "",
            ]
              .filter(Boolean)
              .join(" ");

            return (
              <button
                key={`${p.id}-${nudge && isNudging ? nudge : 0}`}
                className={cls}
                style={style}
                aria-label={
                  isFocus
                    ? p.comingSoon
                      ? `${p.title} — yakında`
                      : `${p.title} — projeyi aç`
                    : `${p.title} — bu projeye dön`
                }
                aria-disabled={isFocus && p.comingSoon ? true : undefined}
                tabIndex={isBack ? -1 : 0}
                onClick={() => {
                  if (isFocus) {
                    openFocused();
                  } else {
                    setFocus(i);
                  }
                }}
              >
                <div className="card-head">
                  <span className="card-num">{p.number}</span>
                  <span className="card-cat">{p.category}</span>
                </div>
                <div className="card-body">
                  <h2 className="card-title">{p.title}</h2>
                  <p className="card-tag">{p.tagline}</p>
                </div>
                <div className="card-foot">
                  <div className="card-rule" aria-hidden="true" />
                  <div className="card-foot-row">
                    <span>{p.comingSoon ? "— yakında" : `:${p.port}`}</span>
                    <span className="card-cta">
                      {p.comingSoon ? "Ön İzleme" : "Projeyi Aç"}{" "}
                      <span className="arrow">{p.comingSoon ? "◷" : "→"}</span>
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        <footer className="rotunda-footer">
          <div className="rotunda-hint">
            <kbd>←</kbd>
            <kbd>→</kbd>
            <span>gez</span>
            <kbd>Enter</kbd>
            <span>aç</span>
          </div>
          <div className="rotunda-dots" role="tablist" aria-label="Projeler">
            {PROJECTS.map((p, i) => (
              <button
                key={p.id}
                className={`rotunda-dot ${i === focus ? "is-on" : ""}`}
                role="tab"
                aria-selected={i === focus}
                aria-label={`${i + 1}. proje: ${p.title}`}
                onClick={() => setFocus(i)}
              />
            ))}
          </div>
          <div className="rotunda-launcher">
            Odak · <em>{focused.title}</em>
            {focused.comingSoon ? <span className="soon-tag">yakında</span> : null}
          </div>
        </footer>
      </div>
    </div>
  );
}
