const API_ROOT = window.location.origin;
let startupUpdateCheck;

async function apiRequest(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_ROOT}${path}`, options);
  } catch (error) {
    throw new Error(
      `The local service is unavailable. Start the app's backend and try again. (${error.message})`
    );
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the HTTP status message if the server did not return JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

export const api = {
  health: () => apiRequest("/api/health"),
  startupUpdateCheck: () => {
    if (!startupUpdateCheck) startupUpdateCheck = apiRequest("/api/updates/check");
    return startupUpdateCheck;
  },
  checkForUpdates: () => apiRequest("/api/updates/check"),
  installUpdate: (installerUrl) => apiRequest("/api/updates/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ installer_url: installerUrl })
  }),
  uploads: () => apiRequest("/api/uploads"),
  trainingDatasets: () => apiRequest("/api/training-datasets"),
  evaluateTrainingDataset: (trainingDatasetId) => apiRequest("/api/training-datasets/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ training_dataset_id: trainingDatasetId })
  }),
  questionPreferences: () => apiRequest("/api/questions/preferences"),
  rateQuestion: (questionId, rating, generationId = null) => apiRequest("/api/questions/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question_id: questionId,
      rating,
      ...(generationId ? { generation_id: generationId } : {})
    })
  }),
  clearQuestionPreferences: () => apiRequest("/api/questions/feedback", { method: "DELETE" }),
  trainingTopics: (id) => apiRequest(`/api/training-datasets/${encodeURIComponent(id)}/topics`),
  generateQuestions: (payload) => apiRequest("/api/questions/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }),
  saveQuestions: (generationId, questionIds) => apiRequest("/api/questions/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation_id: generationId, question_ids: questionIds })
  }),
  revealQuestionAnswers: (generationId, questionIds) => apiRequest("/api/questions/answers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation_id: generationId, question_ids: questionIds })
  }),
  savedQuestions: (includeAnswers = false, sessionId = null) => apiRequest(
    `/api/questions/saved?include_answers=${includeAnswers}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ""}`
  ),
  renameSavedSession: (id, title) => apiRequest(`/api/questions/sessions/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  }),
  deleteSavedSession: (id) => apiRequest(`/api/questions/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE"
  }),
  latestAnalysis: () => apiRequest("/api/analysis/latest"),
  settings: () => apiRequest("/api/settings"),
  saveSettings: (settings) => apiRequest("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  }),
  uploadMaterial: async (file) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest("/api/upload", { method: "POST", body: form });
  },
  uploadLesson: async (file) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest("/api/lessons/upload", { method: "POST", body: form });
  },
  savedMaterial: (id) => apiRequest(`/api/uploads/${encodeURIComponent(id)}/text`),
  deleteUpload: (id) => apiRequest(`/api/uploads/${encodeURIComponent(id)}`, { method: "DELETE" }),
  importTrainingCsv: (file, name) => {
    const form = new FormData();
    form.append("file", file);
    if (name) form.append("name", name);
    return apiRequest("/api/training-datasets/csv", { method: "POST", body: form });
  },
  createDocumentDataset: (payload) => apiRequest("/api/training-datasets/documents", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }),
  deleteTrainingDataset: (id) => apiRequest(`/api/training-datasets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  predict: async (payload, onProgress) => {
    let response;
    try {
      response = await fetch(`${API_ROOT}/api/predict/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    } catch (error) {
      throw new Error(`The local service is unavailable. (${error.message})`);
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Prediction failed (${response.status})`);
    }
    if (!response.body) throw new Error("The local service did not provide prediction progress.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let result = null;
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = pending.split("\n\n");
      pending = frames.pop() || "";
      for (const frame of frames) {
        const event = frame.match(/^event:\s*(.+)$/m)?.[1];
        const data = frame.match(/^data:\s*(.+)$/m)?.[1];
        if (!event || !data) continue;
        const parsed = JSON.parse(data);
        if (event === "progress") onProgress(parsed);
        if (event === "error") throw new Error(parsed.detail || "Prediction failed.");
        if (event === "result") result = parsed;
      }
      if (done) break;
    }
    if (!result) throw new Error("The prediction stream ended without a result.");
    return result;
  }
};
