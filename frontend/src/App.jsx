import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import ConfigScreen from "./components/ConfigScreen.jsx";
import ExperienceScreen from "./components/ExperienceScreen.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const normalizePrompt = (text) => (text || "").replace(/\s+/g, " ").trim().toLowerCase();

const api = axios.create({
  baseURL: API_BASE_URL || undefined,
});

const App = () => {
  const [phase, setPhase] = useState("config");
  const [sessionId, setSessionId] = useState(null);
  const [story, setStory] = useState([]);
  const [stepCount, setStepCount] = useState(0);
  const [maxSteps, setMaxSteps] = useState(10);
  const [hasRemainingSteps, setHasRemainingSteps] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [globalError, setGlobalError] = useState(null);
  const [configSnapshot, setConfigSnapshot] = useState(null);
  const [defaultConfig, setDefaultConfig] = useState(null);

  useEffect(() => {
    const fetchDefaults = async () => {
      try {
        const { data } = await api.get("/api/default-config");
        setDefaultConfig(data);
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn("Failed to load default config", error);
      }
    };

    fetchDefaults();
  }, []);

  const handleConfigSubmit = async (config) => {
    setGlobalError(null);

    const trimmedKey = (config.apiKey || "").trim();
    const normalizedBasePrompt = normalizePrompt(config.basePrompt);
    const prebakedMatch =
      defaultConfig?.prebakedPresets?.find((preset) => {
        if (!normalizedBasePrompt) return false;
        if (config._prebakedSlug && preset.slug === config._prebakedSlug) {
          return true;
        }
        return preset.normalizedBasePrompt === normalizedBasePrompt;
      }) || null;

    if (!trimmedKey && (!prebakedMatch || !prebakedMatch.hasRootVideo)) {
      setGlobalError(
        "Provide an OpenAI API key or select a preset that includes prebaked scenes."
      );
      return;
    }

    setIsSubmitting(true);

    try {
      const payload = {
        apiKey: trimmedKey,
        plannerModel: config.plannerModel,
        soraModel: config.soraModel,
        videoSize: config.videoSize,
        basePrompt: config.basePrompt,
      };
      const { data } = await api.post("/api/session", payload);
      setSessionId(data.sessionId);
      setStory(data.story);
      setStepCount(data.stepCount);
      setMaxSteps(data.maxSteps);
      setHasRemainingSteps(data.hasRemainingSteps);
      setConfigSnapshot({
        ...config,
        apiKey: trimmedKey,
        _prebakedSlug: prebakedMatch?.slug || null,
      });
      setPhase("experience");
    } catch (error) {
      const message = error.response?.data?.detail || error.message || "Failed to start session.";
      setGlobalError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChoice = async (choiceIndex) => {
    if (!sessionId) return;
    setIsGenerating(true);
    setGlobalError(null);

    try {
      const { data } = await api.post(`/api/session/${sessionId}/choice`, { choiceIndex });
      setStory(data.story);
      setStepCount(data.stepCount);
      setMaxSteps(data.maxSteps);
      setHasRemainingSteps(data.hasRemainingSteps);
    } catch (error) {
      const message = error.response?.data?.detail || error.message || "Failed to advance story.";
      setGlobalError(message);
    } finally {
      setIsGenerating(false);
    }
  };

  const context = useMemo(
    () => ({
      sessionId,
      configSnapshot,
      story,
      stepCount,
      maxSteps,
      hasRemainingSteps,
    }),
    [sessionId, configSnapshot, story, stepCount, maxSteps, hasRemainingSteps]
  );

  return phase === "config" ? (
    <ConfigScreen
      onSubmit={handleConfigSubmit}
      isSubmitting={isSubmitting}
      error={globalError}
      apiBaseUrl={API_BASE_URL}
      defaultConfig={defaultConfig}
    />
  ) : (
    <ExperienceScreen
      context={context}
      onMakeChoice={handleChoice}
      isGenerating={isGenerating}
      error={globalError}
      apiBaseUrl={API_BASE_URL}
      onRestart={() => {
        setPhase("config");
        setSessionId(null);
        setStory([]);
        setStepCount(0);
        setMaxSteps(10);
        setHasRemainingSteps(true);
        setConfigSnapshot(null);
        setGlobalError(null);
      }}
    />
  );
};

export default App;
