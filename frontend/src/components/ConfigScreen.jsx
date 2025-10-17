import { useEffect, useMemo, useState } from "react";
import "../styles/config.css";

const PRESETS = [
  {
    label: "Neon Midnight Walk",
    slug: "neon_midnight_walk",
    basePrompt:
      "Photorealistic rainy midnight street in a near-future Tokyo district. Camera glides beside a lone figure walking past noodle stalls, holographic billboards, flickering neon reflections in puddles, steam rising from manholes, and curious onlookers in reflective raincoats. Emphasize cinematic lighting, wet asphalt textures, and the sense that anything could emerge from the crowd.",
  },
  {
    label: "Verdant Quest",
    slug: "verdant_quest",
    basePrompt:
      "Photorealistic enchanted forest adventure at golden hour. Follow an explorer in weathered travel gear trekking through towering moss-covered trees, shafts of light cutting through mist, ancient stone ruins hidden under vines, and distant drumbeats hinting at hidden civilizations. The air feels alive with curiosity and imminent discovery.",
  },
  {
    label: "House of Echoes",
    slug: "house_of_echoes",
    basePrompt:
      "Photorealistic claustrophobic horror inside a decaying Victorian mansion. The protagonist moves room to room; each doorway reveals a new terror: portraits whose eyes bleed shadows, a nursery of toys that whisper, a dining hall table set for spirits. Lighting is minimal, with handheld flashlight beams and erratic power surges casting unsettling moving silhouettes.",
  },
];

const normalizePrompt = (text) => (text || "").replace(/\s+/g, " ").trim().toLowerCase();

const ConfigScreen = ({ onSubmit, isSubmitting, error, apiBaseUrl, defaultConfig }) => {
  const [form, setForm] = useState({
    apiKey: "",
    plannerModel: "gpt-5-chat",
    soraModel: "sora-2",
    videoSize: "1280x720",
    basePrompt:
      "A cozy fantasy village at dusk, with glowing lanterns, narrow cobblestone streets, and a mysterious whisper about an ancient forest relic.",
  });
  const [localError, setLocalError] = useState(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedKey = window.localStorage.getItem("sora_cyoa_api_key");
    if (storedKey) {
      setForm((prev) => ({ ...prev, apiKey: storedKey }));
    }
  }, []);

  useEffect(() => {
    if (!defaultConfig) return;
    setForm((prev) => ({
      ...prev,
      apiKey: prev.apiKey || defaultConfig.apiKey || "",
      plannerModel: prev.plannerModel || defaultConfig.plannerModel || "gpt-5-chat",
      soraModel: prev.soraModel || defaultConfig.soraModel || "sora-2",
      videoSize: prev.videoSize || defaultConfig.videoSize || "1280x720",
    }));
  }, [defaultConfig]);

  const maskedKey = useMemo(() => {
    if (!form.apiKey) return "";
    return form.apiKey.replace(/.(?=.{4})/g, "·");
  }, [form.apiKey]);

  const prebakedMatch = useMemo(() => {
    if (!defaultConfig?.prebakedPresets?.length) return null;
    const normalized = normalizePrompt(form.basePrompt);
    if (!normalized) return null;
    return (
      defaultConfig.prebakedPresets.find(
        (preset) => preset.normalizedBasePrompt === normalized
      ) || null
    );
  }, [defaultConfig, form.basePrompt]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    setLocalError(null);
    if (name === "apiKey" && typeof window !== "undefined") {
      window.localStorage.setItem("sora_cyoa_api_key", value);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const trimmedKey = (form.apiKey || "").trim();
    const prebakedReady = Boolean(prebakedMatch && prebakedMatch.hasRootVideo);

    if (!trimmedKey && !prebakedReady) {
      setLocalError("Provide an OpenAI API key or pick a preset with prebaked footage.");
      return;
    }

    setLocalError(null);

    const submission = {
      ...form,
      apiKey: trimmedKey,
      _prebakedSlug: prebakedMatch?.slug || null,
      _usesPrebaked: prebakedReady,
    };

    onSubmit(submission);
  };

  return (
    <div className="config-shell">
      <div className="config-backdrop" />
      <div className="config-inner">
        <section className="config-hero">
          <p className="config-tag">Sora Control</p>
          <h1>
            Dial in your <span>story engine</span>
          </h1>
          <p className="config-subtitle">
            First, prime the director. Tune the models, set the vibe, and let the cinematic universe know who holds the reins.
          </p>

          <div className="config-presets">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="config-preset"
                disabled={isSubmitting}
                onClick={() => {
                  setForm((prev) => ({ ...prev, basePrompt: preset.basePrompt }));
                  setLocalError(null);
                }}
              >
                <span>{preset.label}</span>
                <small>Inject prompt</small>
              </button>
            ))}
          </div>

          <div className="config-meta">
            <span>Backend:</span>
            <strong>{apiBaseUrl || "http://localhost:8000"}</strong>
          </div>
        </section>

        <section className="config-panel">
          <div className="panel-glass">
            <header>
              <h2>Launch Configuration</h2>
            </header>

            <form onSubmit={handleSubmit}>
              <label className="field">
                <span>OpenAI API key</span>
                <div className="masked">
                  <input
                    name="apiKey"
                    type="password"
                    placeholder="sk-..."
                    value={form.apiKey}
                    onChange={handleChange}
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <span className="mask-preview">{maskedKey}</span>
                </div>
              </label>

              <div className="field-grid">
                <label className="field">
                  <span>Planner model</span>
                  <input
                    name="plannerModel"
                    value={form.plannerModel}
                    onChange={handleChange}
                    placeholder="gpt-5"
                    required
                  />
                </label>
                <label className="field">
                  <span>Sora model</span>
                  <select name="soraModel" value={form.soraModel} onChange={handleChange}>
                    <option value="sora-2">sora-2</option>
                    <option value="sora-2-pro">sora-2-pro</option>
                  </select>
                </label>
                <label className="field">
                  <span>Video size</span>
                  <select name="videoSize" value={form.videoSize} onChange={handleChange}>
                    <option value="1280x720">1280 × 720</option>
                    <option value="1920x1080">1920 × 1080</option>
                    <option value="720x1280">720 × 1280</option>
                  </select>
                </label>
              </div>

              <label className="field">
                <span>World / tone prompt</span>
                <textarea
                  name="basePrompt"
                  value={form.basePrompt}
                  onChange={handleChange}
                  rows={6}
                  required
                />
              </label>

              {prebakedMatch && (
                <div
                  className={`info-banner ${prebakedMatch.hasRootVideo ? "ready" : "warning"
                    }`}
                >
                  {prebakedMatch.hasRootVideo
                    ? `Prebaked scenes detected: ${prebakedMatch.label}. API key optional.`
                    : `Prebaked preset detected (${prebakedMatch.label}), but source media is missing on disk.`}
                </div>
              )}

              {(localError || error) && (
                <div className="error-banner">{localError || error}</div>
              )}

              <button type="submit" className="launch" disabled={isSubmitting}>
                {isSubmitting ? "Generating opening scene…" : "Launch cinematic adventure"}
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ConfigScreen;
