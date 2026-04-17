import React, { useCallback, useEffect, useMemo, useState } from "react";

type ServiceStatus = "online" | "offline" | "checking";
type FeedTone = "info" | "good" | "warn";

interface ModuleDef {
  id: string;
  label: string;
  tagline: string;
  description: string;
  port: number;
  accent: string;
  surface: string;
  pipeline: string[];
}

interface RuntimeModule extends ModuleDef {
  status: ServiceStatus;
  lastSeen: string | null;
}

interface FeedItem {
  id: number;
  message: string;
  tone: FeedTone;
  time: string;
}

const MODULES: ModuleDef[] = [
  {
    id: "ai-spiker",
    label: "AI Spiker",
    tagline: "Senkronize anlatım üretimi",
    description:
      "Video yükle, takım adlarını gir; YOLOv8 + ByteTrack tespiti, AI anlatım üretimi ve TTS sentezi tek pipeline'da çalışır.",
    port: 5173,
    accent: "oklch(0.77 0.145 72)",
    surface: "oklch(0.77 0.145 72 / 0.08)",
    pipeline: ["Yükleme", "Takip", "Anlatım", "TTS Render"],
  },
  {
    id: "mac-chatbot",
    label: "Maç Chatbot",
    tagline: "Zaman damgalı sohbet",
    description:
      "Maç verisini konuşarak sorgular, yanıtla birlikte videoda ilgili saniyeye otomatik sarar.",
    port: 5000,
    accent: "oklch(0.76 0.105 165)",
    surface: "oklch(0.76 0.105 165 / 0.08)",
    pipeline: ["Context Load", "Prompt", "JSON Reply", "Video Seek"],
  },
  {
    id: "var-engine",
    label: "VAR Engine",
    tagline: "IFAB + RAG karar motoru",
    description:
      "Pozisyonu teknik olarak çözümler, ilgili kuralı getirir ve nihai hakem kararını raporlar.",
    port: 8001,
    accent: "oklch(0.68 0.175 28)",
    surface: "oklch(0.68 0.175 28 / 0.08)",
    pipeline: ["Frame Read", "Rule Retrieve", "Decision", "Evidence"],
  },
];

const ASSET_PRESETS = [
  "test_5dk.mp4",
  "messi.mp4",
  "ronaldo.mp4",
  "ofsayt_pozisyonu.mp4",
  "penalti_pozisyonu.mp4",
];

const HEALTH_CHECK_INTERVAL_MS = 12000;
const HEALTH_CHECK_TIMEOUT_MS = 2400;
const START_SCRIPT_CMD = "cd '/Users/alperburakdogan/Desktop/Bitirme Proje' && ./start_all.sh";

function nowStamp(): string {
  return new Date().toLocaleTimeString("tr-TR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(media.matches);

    const handleChange = (event: MediaQueryListEvent) => {
      setReduced(event.matches);
    };

    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  return reduced;
}

async function pingService(port: number): Promise<boolean> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);

  try {
    await fetch(`http://127.0.0.1:${port}/`, {
      mode: "no-cors",
      signal: controller.signal,
      cache: "no-store",
    });
    return true;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeout);
  }
}

const UnifiedHub: React.FC = () => {
  const reducedMotion = useReducedMotion();

  const [modules, setModules] = useState<RuntimeModule[]>(() =>
    MODULES.map((module) => ({
      ...module,
      status: "checking",
      lastSeen: null,
    })),
  );

  const [activeModuleId, setActiveModuleId] = useState<string>(MODULES[0].id);
  const [selectedAsset, setSelectedAsset] = useState<string>(ASSET_PRESETS[0]);
  const [matchTitle, setMatchTitle] = useState<string>("Fenerbahçe vs Karagümrük Analiz Akışı");
  const [clock, setClock] = useState<string>(() => nowStamp());
  const [feed, setFeed] = useState<FeedItem[]>(() => [
    {
      id: Date.now(),
      message: "Match Studio hazır. Servis kontrolleri başlatıldı.",
      tone: "info",
      time: nowStamp(),
    },
  ]);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [frameRevision, setFrameRevision] = useState(0);
  const [frameReady, setFrameReady] = useState(false);
  const [pulseStep, setPulseStep] = useState(0);

  const activeModule = useMemo(() => {
    return modules.find((module) => module.id === activeModuleId) ?? modules[0];
  }, [activeModuleId, modules]);

  const onlineCount = useMemo(
    () => modules.filter((module) => module.status === "online").length,
    [modules],
  );

  const pushFeed = useCallback((message: string, tone: FeedTone = "info") => {
    setFeed((previous) => {
      const next: FeedItem = {
        id: Date.now() + Math.floor(Math.random() * 999),
        message,
        tone,
        time: nowStamp(),
      };
      return [next, ...previous].slice(0, 9);
    });
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setClock(nowStamp());
    }, 1000);

    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let mounted = true;

    const runCheck = async () => {
      const states = await Promise.all(
        MODULES.map(async (module) => {
          const reachable = await pingService(module.port);
          return {
            ...module,
            status: reachable ? "online" : ("offline" as ServiceStatus),
          };
        }),
      );

      if (!mounted) {
        return;
      }

      setModules((previous) => {
        const previousById = new Map(previous.map((module) => [module.id, module]));

        const nextModules = states.map((module) => {
          const old = previousById.get(module.id);
          const newLastSeen = module.status === "online" ? nowStamp() : old?.lastSeen ?? null;

          if (old && old.status !== module.status) {
            if (module.status === "online") {
              pushFeed(`${module.label} erişilebilir durumda.`, "good");
            } else {
              pushFeed(`${module.label} şu anda kapalı görünüyor.`, "warn");
            }
          }

          return {
            ...module,
            lastSeen: newLastSeen,
          };
        });

        return nextModules;
      });
    };

    runCheck();
    const interval = window.setInterval(runCheck, HEALTH_CHECK_INTERVAL_MS);

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [pushFeed]);

  useEffect(() => {
    setFrameReady(false);
    setPulseStep(0);
  }, [activeModuleId, frameRevision]);

  useEffect(() => {
    const steps = activeModule.pipeline;
    if (steps.length === 0) {
      return;
    }

    const speed = reducedMotion ? 2500 : 1400;
    const interval = window.setInterval(() => {
      setPulseStep((previous) => (previous + 1) % steps.length);
    }, speed);

    return () => window.clearInterval(interval);
  }, [activeModule.pipeline, reducedMotion]);

  const activeModuleUrl = `http://127.0.0.1:${activeModule.port}/`;

  const openActiveModule = () => {
    window.open(activeModuleUrl, "_blank", "noopener,noreferrer");
  };

  const applyContextNote = () => {
    pushFeed(
      `${activeModule.label} için bağlam güncellendi: ${matchTitle} / ${selectedAsset}`,
      "info",
    );
  };

  const copyStartScript = async () => {
    try {
      await navigator.clipboard.writeText(START_SCRIPT_CMD);
      pushFeed("Başlatma komutu panoya kopyalandı.", "good");
    } catch {
      pushFeed("Pano erişimi başarısız. Komutu elle kopyalamayı deneyin.", "warn");
    }
  };

  return (
    <div className="studio-shell">
      <div className="studio-ambient" aria-hidden="true" />
      <div className="studio-noise" aria-hidden="true" />

      <header className="studio-topbar reveal-seq">
        <div className="brand-block">
          <span className="brand-mark">MS</span>
          <div>
            <p className="brand-title">Match Studio</p>
            <p className="brand-subtitle">AI Spiker + Chatbot + VAR birleşik çalışma alanı</p>
          </div>
        </div>

        <div className="topbar-meta">
          <span className="meta-pill">
            <span className={`status-core status-${onlineCount === 0 ? "offline" : "online"}`} />
            {onlineCount}/3 servis erişilebilir
          </span>
          <span className="meta-clock">{clock}</span>
          <button type="button" className="meta-action" onClick={copyStartScript}>
            Başlatma Komutunu Kopyala
          </button>
        </div>
      </header>

      <main className={`studio-main ${isFocusMode ? "is-focus" : ""}`}>
        <aside className="studio-left panel reveal-seq" style={{ ["--delay" as string]: "120ms" }}>
          <section className="panel-section">
            <h2>Modül Kontrolü</h2>
            <div className="module-list">
              {modules.map((module) => {
                const isActive = module.id === activeModuleId;
                return (
                  <button
                    type="button"
                    key={module.id}
                    className={`module-card ${isActive ? "is-active" : ""}`}
                    style={{ ["--module-accent" as string]: module.accent, ["--module-surface" as string]: module.surface }}
                    onClick={() => setActiveModuleId(module.id)}
                  >
                    <div className="module-top">
                      <span className={`status-core status-${module.status}`} />
                      <strong>{module.label}</strong>
                      <span className="module-port">:{module.port}</span>
                    </div>
                    <p className="module-tagline">{module.tagline}</p>
                    <p className="module-description">{module.description}</p>
                    <p className="module-foot">
                      {module.status === "online"
                        ? `Son doğrulama ${module.lastSeen ?? "az önce"}`
                        : module.status === "checking"
                          ? "Erişilebilirlik kontrol ediliyor"
                          : "Servis yanıt vermiyor"}
                    </p>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="panel-section context-panel">
            <h2>Maç Bağlamı</h2>
            <label className="input-label" htmlFor="matchTitleInput">
              Çalışma Başlığı
            </label>
            <input
              id="matchTitleInput"
              className="input-field"
              value={matchTitle}
              onChange={(event) => setMatchTitle(event.target.value)}
              placeholder="Örn: Derbi kritik pozisyon paketi"
            />

            <label className="input-label" htmlFor="assetSelect">
              Aktif Asset
            </label>
            <select
              id="assetSelect"
              className="input-field"
              value={selectedAsset}
              onChange={(event) => setSelectedAsset(event.target.value)}
            >
              {ASSET_PRESETS.map((asset) => (
                <option key={asset} value={asset}>
                  {asset}
                </option>
              ))}
            </select>

            <div className="context-actions">
              <button type="button" className="soft-button" onClick={applyContextNote}>
                Bağlam Notu Ekle
              </button>
              <button type="button" className="soft-button" onClick={openActiveModule}>
                Modülü Yeni Sekmede Aç
              </button>
            </div>
          </section>
        </aside>

        <section className="studio-workspace panel reveal-seq" style={{ ["--delay" as string]: "200ms" }}>
          <div className="workspace-head">
            <div className="workspace-tabs" role="tablist" aria-label="Çalışma alanı sekmeleri">
              {modules.map((module) => {
                const selected = module.id === activeModule.id;
                return (
                  <button
                    key={module.id}
                    role="tab"
                    aria-selected={selected}
                    type="button"
                    className={`workspace-tab ${selected ? "is-selected" : ""}`}
                    style={{ ["--tab-accent" as string]: module.accent }}
                    onClick={() => setActiveModuleId(module.id)}
                  >
                    {module.label}
                  </button>
                );
              })}
            </div>

            <div className="workspace-actions">
              <button type="button" className="ghost-button" onClick={() => setFrameRevision((value) => value + 1)}>
                Yenile
              </button>
              <button type="button" className="ghost-button" onClick={() => setIsFocusMode((value) => !value)}>
                {isFocusMode ? "Paneli Geri Getir" : "Odak Modu"}
              </button>
            </div>
          </div>

          <div
            className="workspace-stage"
            style={{ ["--active-accent" as string]: activeModule.accent, ["--active-surface" as string]: activeModule.surface }}
          >
            <div className="stage-topline">
              <div>
                <p className="stage-title">{activeModule.label}</p>
                <p className="stage-subtitle">{activeModule.tagline}</p>
              </div>
              <span className={`stage-status stage-status-${activeModule.status}`}>
                {activeModule.status === "online"
                  ? "Canlı"
                  : activeModule.status === "checking"
                    ? "Kontrol"
                    : "Çevrimdışı"}
              </span>
            </div>

            <div className="stage-frame-shell">
              <iframe
                key={`${activeModule.id}-${frameRevision}`}
                className={`stage-frame ${frameReady ? "is-ready" : ""}`}
                src={activeModuleUrl}
                title={`${activeModule.label} çalışma alanı`}
                onLoad={() => setFrameReady(true)}
              />

              {activeModule.status === "offline" && (
                <div className="offline-overlay">
                  <p>{activeModule.label} servisi şu an erişilebilir değil.</p>
                  <button type="button" onClick={openActiveModule}>
                    Modülü Ayrı Sekmede Aç
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="pipeline-strip" style={{ ["--strip-accent" as string]: activeModule.accent }}>
            {activeModule.pipeline.map((step, index) => (
              <div
                key={step}
                className={`pipeline-step ${index === pulseStep ? "is-active" : ""}`}
                style={{ ["--step-index" as string]: `${index}` }}
              >
                <span>{step}</span>
              </div>
            ))}
          </div>
        </section>

        <aside className="studio-right panel reveal-seq" style={{ ["--delay" as string]: "280ms" }}>
          <section className="panel-section status-section">
            <h2>Canlı Durum</h2>
            <div className="status-grid">
              <article>
                <h3>Servis Sağlığı</h3>
                <p>{onlineCount}/3 çevrimiçi</p>
              </article>
              <article>
                <h3>Aktif Modül</h3>
                <p>{activeModule.label}</p>
              </article>
              <article>
                <h3>Seçili Asset</h3>
                <p>{selectedAsset}</p>
              </article>
            </div>
          </section>

          <section className="panel-section pulse-section">
            <h2>Pipeline Pulse</h2>
            <div className="pulse-list" style={{ ["--pulse-accent" as string]: activeModule.accent }}>
              {activeModule.pipeline.map((step, index) => (
                <div key={`${step}-pulse`} className={`pulse-row ${index === pulseStep ? "is-active" : ""}`}>
                  <span>{step}</span>
                  <span className="pulse-dot" />
                </div>
              ))}
            </div>
          </section>

          <section className="panel-section feed-section">
            <h2>Olay Akışı</h2>
            <ul className="feed-list">
              {feed.map((item) => (
                <li key={item.id} className={`feed-item tone-${item.tone}`}>
                  <span className="feed-time">{item.time}</span>
                  <p>{item.message}</p>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </main>
    </div>
  );
};

export default UnifiedHub;
