import { useEffect, useRef, useState } from "react";
import {
  AlertCircle, BrainCircuit, Check, ChevronDown, ChevronRight, Database, Download, Eye,
  FileSpreadsheet, FileText, FileUp, FolderOpen, Menu, Printer, RotateCcw,
  NotebookText, Pencil, Save, ShieldCheck, Sparkles, ThumbsDown, ThumbsUp, Trash2, Upload, X
} from "lucide-react";
import Sidebar from "./components/Sidebar.jsx";
import { api } from "./lib/api.js";

const initialSettings = { language: "English", appearance: "orange" };
const initialQuestionForm = {
  topic: "All Topics",
  question_type: "Multiple Choice",
  difficulty: "Moderate",
  number_of_questions: 10,
  questions_per_topic: 3,
  similarity_threshold: 82,
  weights: { historical_frequency: 40, course_importance: 30, model_confidence: 20, recent_occurrence: 10 }
};

export default function App() {
  const [page, setPage] = useState("Upload");
  const [menuOpen, setMenuOpen] = useState(false);
  const [settings, setSettings] = useState(initialSettings);
  const [uploads, setUploads] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("builtin");
  const [trainingDatasetName, setTrainingDatasetName] = useState("");
  const [selectedTrainingUploads, setSelectedTrainingUploads] = useState([]);
  const [trainingLabels, setTrainingLabels] = useState({});
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [queuedFiles, setQueuedFiles] = useState([]);
  const [latest, setLatest] = useState(null);
  const [progress, setProgress] = useState([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [appVersion, setAppVersion] = useState("");
  const [updateOffer, setUpdateOffer] = useState(null);
  const [updateCheckBusy, setUpdateCheckBusy] = useState(false);
  const [updateState, setUpdateState] = useState("");
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [questionDatasetId, setQuestionDatasetId] = useState("builtin");
  const [questionUploadIds, setQuestionUploadIds] = useState([]);
  const [questionTopics, setQuestionTopics] = useState([]);
  const [lessonUploads, setLessonUploads] = useState([]);
  const [lessonUploadBusy, setLessonUploadBusy] = useState(false);
  const [lessonProgress, setLessonProgress] = useState(null);
  const [useExamStyle, setUseExamStyle] = useState(true);
  const [savedQuestions, setSavedQuestions] = useState([]);
  const [savedQuestionsBusy, setSavedQuestionsBusy] = useState(false);
  const [expandedSavedSessions, setExpandedSavedSessions] = useState([]);
  const [savedAnswerSessions, setSavedAnswerSessions] = useState([]);
  const [savedAnswersLoadedSessions, setSavedAnswersLoadedSessions] = useState([]);
  const [questionResult, setQuestionResult] = useState(null);
  const [questionPreferences, setQuestionPreferences] = useState({
    ratings_recorded: 0,
    positive_ratings: 0,
    negative_ratings: 0,
    learned_preferences: []
  });
  const [questionAnswers, setQuestionAnswers] = useState({});
  const [questionBusy, setQuestionBusy] = useState(false);
  const [questionsSaved, setQuestionsSaved] = useState(false);
  const [questionForm, setQuestionForm] = useState({
    ...initialQuestionForm,
    weights: { ...initialQuestionForm.weights }
  });
  const fileInput = useRef(null);
  const trainingFileInput = useRef(null);
  const lessonFileInput = useRef(null);
  const csvInput = useRef(null);
  const knownQuestionUploadIds = useRef(new Set());

  const syncSavedUploads = (savedUploads, excludedNewUploadIds = []) => {
    const availableIds = new Set(savedUploads.map((upload) => upload.id));
    const excludedIds = new Set(excludedNewUploadIds);
    const newlyAddedIds = savedUploads
      .filter((upload) => (
        !knownQuestionUploadIds.current.has(upload.id) && !excludedIds.has(upload.id)
      ))
      .map((upload) => upload.id);
    knownQuestionUploadIds.current = availableIds;
    setUploads(savedUploads);
    setQuestionUploadIds((current) => {
      const selectedIds = new Set([
        ...current.filter((id) => availableIds.has(id)),
        ...newlyAddedIds
      ]);
      return savedUploads
        .filter((upload) => selectedIds.has(upload.id))
        .map((upload) => upload.id);
    });
  };

  useEffect(() => {
    let active = true;
    Promise.all([api.health(), api.settings(), api.uploads(), api.latestAnalysis(), api.trainingDatasets()])
      .then(([health, savedSettings, savedUploads, savedAnalysis, savedDatasets]) => {
        if (!active) return;
        setAppVersion(health.version);
        setSettings({
          language: savedSettings.language || initialSettings.language,
          appearance: savedSettings.appearance || initialSettings.appearance
        });
        syncSavedUploads(savedUploads);
        const savedQuestionSettings = savedSettings.question_generation;
        if (savedQuestionSettings) {
          setQuestionDatasetId(savedQuestionSettings.training_dataset_id || "builtin");
          setQuestionForm({
            ...initialQuestionForm,
            ...savedQuestionSettings,
            similarity_threshold: Math.round(
              (savedQuestionSettings.similarity_threshold ?? 0.82) * 100
            ),
            weights: Object.fromEntries(
              Object.entries({
                ...initialQuestionForm.weights,
                ...Object.fromEntries(
                  Object.entries(savedQuestionSettings.priority_weights || {}).map(
                    ([key, value]) => [key, Math.round(value * 100)]
                  )
                )
              })
            )
          });
          const savedUploadIds = (savedQuestionSettings.upload_ids || []).filter((id) =>
            savedUploads.some((upload) => upload.id === id)
          );
          const allCurrentUploadsWereSelected = savedUploads.length > 0
            && savedUploads.every((upload) => savedUploadIds.includes(upload.id));
          setQuestionUploadIds(
            savedQuestionSettings.upload_ids === null || allCurrentUploadsWereSelected
              ? savedUploads.map((upload) => upload.id)
              : savedUploadIds
          );
        } else {
          setQuestionUploadIds(savedUploads.map((upload) => upload.id));
        }
        setLatest(savedAnalysis);
        setDatasets(savedDatasets);
        if (savedAnalysis?.training_dataset_id
            && savedDatasets.some((dataset) => dataset.id === savedAnalysis.training_dataset_id)) {
          setSelectedDatasetId(savedAnalysis.training_dataset_id);
        }
      })
      .catch((loadError) => {
        if (active) setError(loadError.message);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    api.startupUpdateCheck()
      .then((update) => {
        if (active && update.available) setUpdateOffer(update);
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    api.questionPreferences()
      .then((preferences) => {
        if (active) setQuestionPreferences(preferences);
      })
      .catch((loadError) => {
        if (active) setError(`Question preferences could not be loaded: ${loadError.message}`);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (page !== "Settings") return;
    let active = true;
    api.trainingTopics(questionDatasetId)
      .then(({ topics }) => {
        if (!active) return;
        setQuestionTopics(topics);
        setQuestionForm((current) => ({
          ...current,
          topic: current.topic === "All Topics" || topics.includes(current.topic)
            ? current.topic : "All Topics"
        }));
      })
      .catch((loadError) => {
        if (active) setError(`Training topics could not be loaded: ${loadError.message}`);
      });
    return () => { active = false; };
  }, [page, questionDatasetId]);

  useEffect(() => {
    if (page !== "Saved Questions") return;
    let active = true;
    setSavedQuestionsBusy(true);
    setError("");
    api.savedQuestions()
      .then(({ sessions }) => {
        if (!active) return;
        setSavedQuestions(sessions);
        setExpandedSavedSessions(sessions.length ? [sessions[0].id] : []);
        setSavedAnswerSessions([]);
        setSavedAnswersLoadedSessions([]);
      })
      .catch((loadError) => {
        if (active) setError(`Saved questions could not be loaded: ${loadError.message}`);
      })
      .finally(() => {
        if (active) setSavedQuestionsBusy(false);
      });
    return () => { active = false; };
  }, [page]);

  useEffect(() => {
    document.documentElement.dataset.appearance = settings.appearance;
    document.title = "Clinthinker©";
    const themeColors = { orange: "#100d0a", pink: "#fff8fc", regal: "#171225" };
    document.querySelector('meta[name="theme-color"]')?.setAttribute(
      "content",
      themeColors[settings.appearance] || themeColors.orange
    );
  }, [settings.appearance]);

  const addFiles = (fileList) => {
    const selected = Array.from(fileList || []);
    const accepted = selected.filter((file) => /\.(pdf|docx|txt)$/i.test(file.name));
    if (accepted.length !== selected.length) {
      setError("Only PDF, DOCX, and TXT course materials are supported.");
    } else {
      setError("");
    }
    if (!accepted.length) return;
    setQueuedFiles((current) => {
      const fresh = accepted.filter((file) => !current.some(
        (item) => item.name === file.name && item.size === file.size
      ));
      return [...current, ...fresh.map((file) => ({
        id: crypto.randomUUID(),
        name: file.name,
        bytes: file.size,
        file,
        saved: false
      }))];
    });
  };

  const includeSavedFile = async (upload) => {
    if (queuedFiles.some((item) => item.savedId === upload.id)) return;
    setError("");
    try {
      const savedFile = await api.savedMaterial(upload.id);
      setQueuedFiles((current) => current.some((item) => item.savedId === upload.id)
        ? current
        : [...current, {
          id: crypto.randomUUID(),
          savedId: upload.id,
          name: savedFile.name,
          bytes: upload.bytes,
          text: savedFile.text,
          saved: true
        }]);
    } catch (loadError) {
      setError(`Could not load ${upload.name}: ${loadError.message}`);
    }
  };

  const removeUpload = async (upload) => {
    if (!window.confirm(`Delete ${upload.name} from this device?`)) return;
    setError("");
    try {
      await api.deleteUpload(upload.id);
      setUploads((current) => current.filter((item) => item.id !== upload.id));
      if (lessonUploads.some((lesson) => lesson.id === upload.id)) {
        const remainingLessons = lessonUploads.filter((lesson) => lesson.id !== upload.id);
        setLessonUploads(remainingLessons);
        setQuestionResult(null);
        setQuestionAnswers({});
        setQuestionsSaved(false);
        if (remainingLessons.length) {
          await generateQuestions({ lessons: remainingLessons });
        } else {
          setLessonProgress(null);
        }
      }
      knownQuestionUploadIds.current.delete(upload.id);
      setQueuedFiles((current) => current.filter((item) => item.savedId !== upload.id));
      setQuestionUploadIds((current) => current.filter((id) => id !== upload.id));
      setSelectedTrainingUploads((current) => current.filter((id) => id !== upload.id));
      setQuestionUploadIds((current) => current.filter((id) => id !== upload.id));
      setTrainingLabels((current) => {
        const next = { ...current };
        delete next[upload.id];
        return next;
      });
      setNotice("File deleted from local storage.");
    } catch (deleteError) {
      setError(`Could not delete ${upload.name}: ${deleteError.message}`);
    }
  };

  const refreshTrainingDatasets = async () => {
    const savedDatasets = await api.trainingDatasets();
    setDatasets(savedDatasets);
    return savedDatasets;
  };

  const checkForUpdates = async () => {
    setUpdateCheckBusy(true);
    setUpdateState("");
    try {
      const update = await api.checkForUpdates();
      setUpdateOffer(update.available ? update : null);
    } catch {
      setUpdateOffer(null);
    } finally {
      setUpdateCheckBusy(false);
    }
  };

  const installUpdate = async () => {
    if (!updateOffer) return;
    setUpdateState("downloading");
    try {
      await api.installUpdate(updateOffer.installer_url);
      window.close();
    } catch {
      setUpdateState("failed");
    }
  };

  const uploadTrainingFiles = async (fileList) => {
    const selected = Array.from(fileList || []);
    const accepted = selected.filter((file) => /\.(pdf|docx|txt)$/i.test(file.name));
    if (accepted.length !== selected.length) {
      setError("Training documents must be PDF, DOCX, or TXT files.");
      return;
    }
    setTrainingBusy(true);
    setError("");
    setNotice("");
    try {
      for (const file of accepted) await api.uploadMaterial(file);
      syncSavedUploads(await api.uploads());
      setNotice(`${accepted.length} document${accepted.length === 1 ? "" : "s"} saved for labeling.`);
    } catch (uploadError) {
      setError(`Training document upload failed: ${uploadError.message}`);
      try {
        syncSavedUploads(await api.uploads());
      } catch (refreshError) {
        setError((current) => `${current} Saved-file list could not refresh: ${refreshError.message}`);
      }
    } finally {
      setTrainingBusy(false);
    }
  };

  const importTrainingCsv = async (file) => {
    if (!file) return;
    setTrainingBusy(true);
    setError("");
    setNotice("");
    try {
      const dataset = await api.importTrainingCsv(file, trainingDatasetName.trim());
      await refreshTrainingDatasets();
      setSelectedDatasetId(dataset.id);
      setQuestionDatasetId(dataset.id);
      setTrainingDatasetName("");
      setNotice(`Imported ${dataset.example_count} labeled rows across ${dataset.topic_count} topics.`);
    } catch (importError) {
      setError(`CSV dataset could not be imported: ${importError.message}`);
    } finally {
      setTrainingBusy(false);
    }
  };

  const createDocumentTrainingDataset = async () => {
    const examples = selectedTrainingUploads.map((uploadId) => ({
      upload_id: uploadId,
      topic: (trainingLabels[uploadId] || "").trim()
    }));
    if (examples.length < 2 || new Set(examples.map((item) => item.topic).filter(Boolean)).size < 2) {
      setError("Choose at least two documents and assign at least two different topic labels.");
      return;
    }
    setTrainingBusy(true);
    setError("");
    setNotice("");
    try {
      const dataset = await api.createDocumentDataset({
        name: trainingDatasetName.trim(),
        examples
      });
      await refreshTrainingDatasets();
      setSelectedDatasetId(dataset.id);
      setQuestionDatasetId(dataset.id);
      setTrainingDatasetName("");
      setSelectedTrainingUploads([]);
      setNotice(`Created training dataset with ${dataset.example_count} labeled documents.`);
    } catch (createError) {
      setError(`Document dataset could not be created: ${createError.message}`);
    } finally {
      setTrainingBusy(false);
    }
  };

  const removeTrainingDataset = async (dataset) => {
    if (!window.confirm(`Delete the training dataset "${dataset.name}"? This does not delete your uploaded files.`)) return;
    setTrainingBusy(true);
    setError("");
    try {
      await api.deleteTrainingDataset(dataset.id);
      const remaining = await refreshTrainingDatasets();
      if (selectedDatasetId === dataset.id) setSelectedDatasetId("builtin");
      if (questionDatasetId === dataset.id) {
        setQuestionDatasetId("builtin");
        setQuestionForm((current) => ({ ...current, topic: "All Topics" }));
      }
      setNotice(`Training dataset "${dataset.name}" deleted.`);
      return remaining;
    } catch (deleteError) {
      setError(`Training dataset could not be deleted: ${deleteError.message}`);
    } finally {
      setTrainingBusy(false);
    }
  };

  const analyze = async () => {
    if (busy) return;
    if (!queuedFiles.length) {
      setError("Upload at least one PDF, DOCX, or TXT file to begin.");
      return;
    }
    setError("");
    setNotice("");
    setBusy(true);
    setProgress([]);
    try {
      const materials = [];
      for (const item of queuedFiles) {
        if (item.saved) {
          materials.push(item);
          continue;
        }
        const uploaded = await api.uploadMaterial(item.file);
        const savedItem = { ...item, saved: true, savedId: uploaded.id, text: uploaded.text };
        materials.push(savedItem);
        setQueuedFiles((current) => current.map(
          (queued) => queued.id === item.id ? savedItem : queued
        ));
      }
      syncSavedUploads(await api.uploads());
      setQueuedFiles(materials);
      const text = materials.map((item) => item.text).filter(Boolean).join("\n\n");
      if (!text.trim()) throw new Error("No readable text was found in the selected files.");
      const result = await api.predict({ text, training_dataset_id: selectedDatasetId }, (event) => {
        setProgress((current) => {
          const order = ["extraction", "cleaning", "features", "analysis"];
          const index = order.indexOf(event.phase);
          const phases = [
            ["extraction", "Text extraction complete"],
            ["cleaning", "Text cleaning complete"],
            ["features", "Feature extraction complete"],
            ["analysis", "Topic analysis"]
          ];
          return phases.map(([phase, label], phaseIndex) => ({
            phase,
            label,
            state: phaseIndex < index || (phaseIndex === index && phase !== "analysis")
              ? "done"
              : phaseIndex === index ? "active" : "pending"
          }));
        });
      });
      setLatest(result);
      setProgress((current) => current.map((item) => ({ ...item, state: "done" })));
      setNotice("Analysis finished. Your result is saved on this device.");
    } catch (analysisError) {
      setError(`Analysis could not be completed: ${analysisError.message}`);
      setProgress([]);
      try {
        setUploads(await api.uploads());
      } catch (refreshError) {
        setError((current) => `${current} Uploaded-file list could not refresh: ${refreshError.message}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const saveSettings = async () => {
    setSettingsBusy(true);
    setError("");
    try {
      const saved = await api.saveSettings({
        language: settings.language,
        appearance: settings.appearance,
        question_generation: {
          training_dataset_id: questionDatasetId,
          upload_ids: uploads.every((upload) => questionUploadIds.includes(upload.id))
            ? null
            : questionUploadIds,
          topic: questionForm.topic,
          question_type: questionForm.question_type,
          difficulty: questionForm.difficulty,
          number_of_questions: Number(questionForm.number_of_questions),
          questions_per_topic: Number(questionForm.questions_per_topic),
          similarity_threshold: Number(questionForm.similarity_threshold) / 100,
          priority_weights: Object.fromEntries(
            Object.entries(questionForm.weights).map(([key, value]) => [key, Number(value) / 100])
          )
        }
      });
      setSettings({
        language: saved.language || initialSettings.language,
        appearance: saved.appearance || initialSettings.appearance
      });
      setNotice("Preferences and question-generation settings saved locally.");
    } catch (saveError) {
      setError(`Preferences could not be saved: ${saveError.message}`);
    } finally {
      setSettingsBusy(false);
    }
  };

  const generateQuestions = async ({
    lessons: lessonOverride = lessonUploads,
    useStyle = useExamStyle
  } = {}) => {
    setError("");
    setNotice("");
    const selectedLessons = lessonOverride.filter((lesson) => lesson.status === "Ready");
    const useLesson = selectedLessons.length > 0;
    const uploadIds = useLesson
      ? selectedLessons.map((lesson) => lesson.id)
      : questionUploadIds;
    if (!uploadIds.length) {
      setError("Select at least one saved course material as a grounded source.");
      return null;
    }
    setQuestionBusy(true);
    if (useLesson) {
      setLessonProgress({
        phase: "generating",
        message: `Generating questions from ${selectedLessons.length} lesson${selectedLessons.length === 1 ? "" : "s"}...`
      });
    }
    setQuestionAnswers({});
    setQuestionsSaved(false);
    try {
      const result = await api.generateQuestions({
        training_dataset_id: questionDatasetId,
        upload_ids: uploadIds,
        lesson_upload_id: useLesson && selectedLessons.length === 1
          ? selectedLessons[0].id
          : null,
        lesson_upload_ids: useLesson ? uploadIds : [],
        lesson_only: useLesson,
        use_exam_style: useStyle,
        topic: questionForm.topic,
        question_type: questionForm.question_type,
        difficulty: questionForm.difficulty,
        number_of_questions: Number(questionForm.number_of_questions),
        questions_per_topic: Number(questionForm.questions_per_topic),
        similarity_threshold: Number(questionForm.similarity_threshold) / 100,
        priority_weights: Object.fromEntries(
          Object.entries(questionForm.weights).map(([key, value]) => [key, Number(value) / 100])
        )
      });
      setQuestionResult(result);
      if (useLesson) {
        setNotice(
          `Generated ${result.questions.length} of ${result.requested_count} requested. Skipped: `
          + `${result.diagnostics.skipped_low_confidence || 0} low-confidence, `
          + `${result.diagnostics.no_valid_distractors || 0} no valid distractors.`
        );
      } else if (result.shortfall > 0) {
        setNotice(
          `Generated ${result.questions.length} supported question${result.questions.length === 1 ? "" : "s"}`
          + `; ${result.shortfall} of `
          + `${result.requested_count} requested could not be supported by the selected source.`
        );
      } else if (result.questions.length) {
        setNotice(`${result.questions.length} supported practice question${result.questions.length === 1 ? "" : "s"} generated.`);
      } else {
        setNotice("");
      }
      if (useLesson) {
        setLessonProgress({
          phase: "done",
          message: `Done — ${result.questions.length} question${result.questions.length === 1 ? "" : "s"} generated from ${selectedLessons.length} lesson${selectedLessons.length === 1 ? "" : "s"}.`
        });
      }
      return result;
    } catch (generationError) {
      setError(`Questions could not be generated: ${generationError.message}`);
      if (useLesson) {
        setLessonProgress({
          phase: "error",
          message: `Question generation from ${selectedLessons.length} lesson${selectedLessons.length === 1 ? "" : "s"} did not complete.`
        });
      }
      return null;
    } finally {
      setQuestionBusy(false);
    }
  };

  const uploadLessons = async (fileList) => {
    const selectedFiles = Array.from(fileList || []);
    if (!selectedFiles.length) return;
    const accepted = selectedFiles.filter((file) => /\.(pdf|docx|txt)$/i.test(file.name));
    const rejected = selectedFiles.filter((file) => !accepted.includes(file));
    const rejectedLessons = rejected.map((file) => ({
      id: `error-${crypto.randomUUID()}`,
      name: file.name,
      bytes: file.size,
      status: "Error",
      error: "Unsupported file type"
    }));
    if (rejectedLessons.length) {
      setLessonUploads((current) => [...current, ...rejectedLessons]);
    }
    if (!accepted.length) {
      setError("Lessons must be PDF, DOCX, or TXT files.");
      return;
    }
    setLessonUploadBusy(true);
    setError("");
    setNotice("");
    setQuestionResult(null);
    setQuestionAnswers({});
    setQuestionsSaved(false);
    let nextLessons = [...lessonUploads, ...rejectedLessons];
    const uploadErrors = rejected.map((file) => `${file.name}: unsupported file type`);
    for (const file of accepted) {
      const pendingId = `pending-${crypto.randomUUID()}`;
      const pendingLesson = {
        id: pendingId,
        name: file.name,
        bytes: file.size,
        status: "Uploading"
      };
      nextLessons = [...nextLessons, pendingLesson];
      setLessonUploads(nextLessons);
      setLessonProgress({
        phase: "uploading",
        message: `Uploading ${file.name}...`
      });
      try {
        const savedLesson = await api.uploadLesson(file);
        nextLessons = nextLessons.map((lesson) => (
          lesson.id === pendingId
            ? { ...lesson, status: "Reading" }
            : lesson
        ));
        setLessonUploads(nextLessons);
        setLessonProgress({
          phase: "reading",
          message: `Reading ${file.name} — ${savedLesson.text_chars.toLocaleString()} characters extracted.`
        });
        await new Promise((resolve) => window.requestAnimationFrame(resolve));
        const readyLesson = {
          id: savedLesson.id,
          name: savedLesson.name,
          bytes: savedLesson.bytes,
          status: "Ready"
        };
        const priorIndex = nextLessons.findIndex(
          (lesson) => lesson.id === savedLesson.id
        );
        nextLessons = nextLessons.filter((lesson) => lesson.id !== pendingId);
        if (priorIndex >= 0 && savedLesson.id !== pendingId) {
          const existingIndex = nextLessons.findIndex(
            (lesson) => lesson.id === savedLesson.id
          );
          if (existingIndex >= 0) nextLessons[existingIndex] = readyLesson;
          else nextLessons.push(readyLesson);
        } else if (!nextLessons.some((lesson) => lesson.id === savedLesson.id)) {
          nextLessons.push(readyLesson);
        }
        setLessonUploads(nextLessons);
      } catch (uploadError) {
        nextLessons = nextLessons.map((lesson) => (
          lesson.id === pendingId
            ? { ...lesson, status: "Error", error: uploadError.message }
            : lesson
        ));
        setLessonUploads(nextLessons);
        uploadErrors.push(`${file.name}: ${uploadError.message}`);
      }
    }
    try {
      const readyLessons = nextLessons.filter((lesson) => lesson.status === "Ready");
      if (readyLessons.length) {
        try {
          syncSavedUploads(
            await api.uploads(),
            readyLessons.map((lesson) => lesson.id)
          );
        } catch (refreshError) {
          uploadErrors.push(`Saved upload list could not refresh: ${refreshError.message}`);
        }
        await generateQuestions({ lessons: readyLessons });
      }
    } finally {
      setLessonUploadBusy(false);
    }
    if (uploadErrors.length) {
      setError(`Some lesson files could not be added: ${uploadErrors.join("; ")}`);
    }
  };

  const removeLessonFromSetup = (lessonId) => {
    const remainingLessons = lessonUploads.filter((lesson) => lesson.id !== lessonId);
    setLessonUploads(remainingLessons);
    setQuestionResult(null);
    setQuestionAnswers({});
    setQuestionsSaved(false);
    if (remainingLessons.some((lesson) => lesson.status === "Ready")) {
      void generateQuestions({ lessons: remainingLessons });
    } else {
      setLessonProgress(null);
    }
  };

  const changeExamStyle = (enabled) => {
    setUseExamStyle(enabled);
    if (lessonUploads.some((lesson) => lesson.status === "Ready")) {
      void generateQuestions({ lessons: lessonUploads, useStyle: enabled });
    }
  };

  const saveGeneratedQuestions = async () => {
    if (!questionResult?.questions?.length) return;
    setQuestionBusy(true);
    setError("");
    try {
      const result = await api.saveQuestions(
        questionResult.generation_id,
        questionResult.questions.map((item) => item.id)
      );
      setQuestionsSaved(true);
      setNotice(`${result.saved_count} question${result.saved_count === 1 ? "" : "s"} saved on this device.`);
    } catch (saveError) {
      setError(`Questions could not be saved: ${saveError.message}`);
    } finally {
      setQuestionBusy(false);
    }
  };

  const revealGeneratedAnswers = async () => {
    if (!questionResult?.questions?.length) return;
    if (Object.keys(questionAnswers).length) {
      setQuestionAnswers({});
      return;
    }
    setQuestionBusy(true);
    setError("");
    try {
      const result = await api.revealQuestionAnswers(
        questionResult.generation_id,
        questionResult.questions.map((item) => item.id)
      );
      setQuestionAnswers(result.answers);
    } catch (answerError) {
      setError(`Answer key could not be loaded: ${answerError.message}`);
    } finally {
      setQuestionBusy(false);
    }
  };

  const rateQuestion = async (item, rating, generationId = null) => {
    setError("");
    try {
      const preferences = await api.rateQuestion(item.id, rating, generationId);
      setQuestionPreferences(preferences);
      setQuestionResult((current) => current && generationId === current.generation_id
        ? {
          ...current,
          questions: current.questions.map((question) => question.id === item.id
            ? { ...question, feedback_rating: rating } : question)
        }
        : current);
      setSavedQuestions((current) => current.map((session) => ({
        ...session,
        questions: session.questions.map((question) => question.id === item.id
          ? { ...question, feedback_rating: rating } : question)
      })));
    } catch (feedbackError) {
      setError(`Question feedback could not be saved: ${feedbackError.message}`);
    }
  };

  const resetQuestionPreferences = async () => {
    if (!window.confirm("Clear all question ratings and reset your global preferences?")) return;
    setError("");
    try {
      setQuestionPreferences(await api.clearQuestionPreferences());
      setQuestionResult((current) => current ? {
        ...current,
        questions: current.questions.map((question) => ({ ...question, feedback_rating: null }))
      } : current);
      setSavedQuestions((current) => current.map((session) => ({
        ...session,
        questions: session.questions.map((question) => ({ ...question, feedback_rating: null }))
      })));
      setNotice("Question preferences were reset.");
    } catch (resetError) {
      setError(`Question preferences could not be reset: ${resetError.message}`);
    }
  };

  const revealSavedAnswers = async (sessionIds = savedQuestions.map((session) => session.id)) => {
    const unloadedSessionIds = sessionIds.filter(
      (sessionId) => !savedAnswersLoadedSessions.includes(sessionId)
    );
    if (unloadedSessionIds.length) {
      setSavedQuestionsBusy(true);
      setError("");
      try {
        const requestedSessionId = unloadedSessionIds.length === 1
          ? unloadedSessionIds[0] : null;
        const { sessions } = await api.savedQuestions(true, requestedSessionId);
        if (requestedSessionId) {
          const requestedSession = sessions.find((session) => session.id === requestedSessionId);
          if (!requestedSession) throw new Error("Saved question session was not found.");
          setSavedQuestions((current) => current.map((session) =>
            session.id === requestedSessionId ? requestedSession : session
          ));
        } else {
          setSavedQuestions(sessions);
        }
        setSavedAnswersLoadedSessions((current) => [
          ...new Set([...current, ...unloadedSessionIds])
        ]);
      } catch (answerError) {
        setError(`Saved answer key could not be loaded: ${answerError.message}`);
        setSavedQuestionsBusy(false);
        return;
      }
      setSavedQuestionsBusy(false);
    }
    setSavedAnswerSessions((current) => [...new Set([...current, ...sessionIds])]);
  };

  const toggleSavedSessionAnswers = (sessionId) => {
    if (savedAnswerSessions.includes(sessionId)) {
      setSavedAnswerSessions((current) => current.filter((id) => id !== sessionId));
      return;
    }
    revealSavedAnswers([sessionId]);
  };

  const toggleAllSavedAnswers = () => {
    const sessionsWithQuestions = savedQuestions
      .filter((session) => session.questions.length)
      .map((session) => session.id);
    const allVisible = sessionsWithQuestions.length > 0
      && sessionsWithQuestions.every((id) => savedAnswerSessions.includes(id));
    if (allVisible) setSavedAnswerSessions([]);
    else revealSavedAnswers(sessionsWithQuestions);
  };

  const renameSavedSession = async (session) => {
    const title = window.prompt("Rename this question session:", session.title);
    if (title === null || !title.trim()) return;
    setError("");
    try {
      const updated = await api.renameSavedSession(session.id, title.trim());
      setSavedQuestions((current) => current.map((item) =>
        item.id === session.id ? { ...item, title: updated.title } : item
      ));
    } catch (renameError) {
      setError(`Session could not be renamed: ${renameError.message}`);
    }
  };

  const deleteSavedSession = async (session) => {
    if (!window.confirm(`Delete "${session.title}" and all ${session.question_count} saved questions in it?`)) return;
    setError("");
    try {
      await api.deleteSavedSession(session.id);
      setSavedQuestions((current) => current.filter((item) => item.id !== session.id));
      setExpandedSavedSessions((current) => current.filter((id) => id !== session.id));
      setSavedAnswerSessions((current) => current.filter((id) => id !== session.id));
      setSavedAnswersLoadedSessions((current) => current.filter((id) => id !== session.id));
    } catch (deleteError) {
      setError(`Session could not be deleted: ${deleteError.message}`);
    }
  };

  const exportQuestionsCsv = () => {
    if (!questionResult?.questions?.length) return;
    const rows = [
      ["Topic", "Type", "Difficulty", "Priority", "Confidence", "Question", "Choices", "Answer", "Source"],
      ...questionResult.questions.map((item) => [
        item.topic, item.question_type, item.difficulty,
        `${Math.round(item.priority_score * 100)}%`,
        `${Math.round(item.confidence_score * 100)}%`,
        item.question_text, item.choices.join(" | "),
        questionAnswers[item.id] || "",
        item.source_reference
      ])
    ];
    const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "clinthinker-practice-questions.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className={`app-shell app-window theme-${settings.appearance}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={(event) => { if (event.currentTarget === event.target) setDragging(false); }}
      onDrop={(event) => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files); }}
    >
      <Sidebar active={page} onNavigate={(next) => {
        setPage(next);
        setMenuOpen(false);
      }} open={menuOpen} onClose={() => setMenuOpen(false)} />
      <main className="main-area medium-main">
        <header className="topbar">
          <button className="menu-trigger" onClick={() => setMenuOpen(true)}><Menu size={18} /><span>MENU</span></button>
          <div className="mobile-brand"><div className="brand-mark"><BrainCircuit size={17} /></div> Clinthinker©</div>
          <div className="breadcrumb"><span>LOCAL WORKSPACE</span><span className="breadcrumb-sep">/</span><strong>{page}</strong></div>
          <div className="local-status"><span /> OFFLINE · SAVED ON THIS DEVICE</div>
        </header>
        {updateOffer && <div className="app-update-banner">
          <span>Version {updateOffer.version} is available</span>
          <button className="button-dark" onClick={installUpdate} disabled={updateState === "downloading"}>
            {updateState === "downloading" ? "Downloading..." : updateState === "failed" ? "Update failed" : "Update"}
          </button>
          <button className="button-outline" disabled={updateState === "downloading"} onClick={() => {
            setUpdateOffer(null);
            setUpdateState("");
          }}>Later</button>
        </div>}

        {page === "Upload" && <UploadPage
          uploads={uploads}
          datasets={datasets}
          selectedDatasetId={selectedDatasetId}
          setSelectedDatasetId={setSelectedDatasetId}
          trainingBusy={trainingBusy}
          queuedFiles={queuedFiles}
          addFiles={addFiles}
          fileInput={fileInput}
          removeQueued={(id) => setQueuedFiles((current) => current.filter((item) => item.id !== id))}
          includeSavedFile={includeSavedFile}
          removeUpload={removeUpload}
          analyze={analyze}
          busy={busy}
          progress={progress}
          latest={latest}
          error={error}
          notice={notice}
          dragging={dragging}
        />}
        {page === "Training Data" && <TrainingDataPage
          uploads={uploads}
          datasets={datasets}
          selectedDatasetId={selectedDatasetId}
          datasetName={trainingDatasetName}
          setDatasetName={setTrainingDatasetName}
          selectedUploads={selectedTrainingUploads}
          setSelectedUploads={setSelectedTrainingUploads}
          labels={trainingLabels}
          setLabels={setTrainingLabels}
          busy={trainingBusy}
          fileInput={trainingFileInput}
          csvInput={csvInput}
          onImportCsv={importTrainingCsv}
          onCreateDocuments={createDocumentTrainingDataset}
          onRemoveUpload={removeUpload}
          onDeleteDataset={removeTrainingDataset}
          error={error}
          notice={notice}
        />}
        {page === "Saved Questions" && <SavedQuestionsPage
          sessions={savedQuestions}
          busy={savedQuestionsBusy}
          expandedSessions={expandedSavedSessions}
          setExpandedSessions={setExpandedSavedSessions}
          visibleAnswerSessions={savedAnswerSessions}
          onToggleAnswers={toggleSavedSessionAnswers}
          onToggleAllAnswers={toggleAllSavedAnswers}
          onRename={renameSavedSession}
          onDelete={deleteSavedSession}
          onRateQuestion={rateQuestion}
          error={error}
        />}
        {page === "Generate Practice Questions" && <PracticeQuestionsPage
          datasets={datasets}
          uploads={uploads}
          selectedDatasetId={questionDatasetId}
          selectedUploadIds={questionUploadIds}
          lessonUploads={lessonUploads}
          lessonUploadBusy={lessonUploadBusy}
          lessonProgress={lessonProgress}
          useExamStyle={useExamStyle}
          onChangeExamStyle={changeExamStyle}
          lessonUploadInput={lessonFileInput}
          onUploadLessons={uploadLessons}
          onRemoveLesson={removeLessonFromSetup}
          form={questionForm}
          onOpenSettings={() => setPage("Settings")}
          result={questionResult}
          preferences={questionPreferences}
          answers={questionAnswers}
          busy={questionBusy}
          saved={questionsSaved}
          onGenerate={generateQuestions}
          onSave={saveGeneratedQuestions}
          onRevealAnswers={revealGeneratedAnswers}
          onExportCsv={exportQuestionsCsv}
          onExportPdf={() => window.print()}
          onRateQuestion={rateQuestion}
          onResetPreferences={resetQuestionPreferences}
          onBack={() => setPage("Upload")}
          error={error}
          notice={notice}
        />}
        {page === "Settings" && <SettingsPage
          settings={settings}
          setSettings={setSettings}
          datasets={datasets}
          uploads={uploads}
          selectedDatasetId={questionDatasetId}
          setSelectedDatasetId={(id) => {
            setQuestionDatasetId(id);
            setQuestionForm((current) => ({ ...current, topic: "All Topics" }));
            setQuestionResult(null);
            setQuestionAnswers({});
          }}
          selectedUploadIds={questionUploadIds}
          setSelectedUploadIds={(update) => {
            setQuestionUploadIds(update);
            setQuestionResult(null);
            setQuestionAnswers({});
          }}
          topics={questionTopics}
          questionForm={questionForm}
          setQuestionForm={(update) => {
            setQuestionForm(update);
            setQuestionResult(null);
            setQuestionAnswers({});
          }}
          saveSettings={saveSettings}
          busy={settingsBusy}
          appVersion={appVersion}
          onCheckForUpdates={checkForUpdates}
          updateCheckBusy={updateCheckBusy}
          error={error}
          notice={notice}
        />}
        {page === "Help" && <HelpPage />}
        <input
          ref={fileInput}
          className="hidden-input"
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          onChange={(event) => { addFiles(event.target.files); event.target.value = ""; }}
        />
        <input
          ref={lessonFileInput}
          className="hidden-input"
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          onChange={(event) => { uploadLessons(event.target.files); event.target.value = ""; }}
        />
        <input
          ref={trainingFileInput}
          className="hidden-input"
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          onChange={(event) => { uploadTrainingFiles(event.target.files); event.target.value = ""; }}
        />
        <input
          ref={csvInput}
          className="hidden-input"
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => { importTrainingCsv(event.target.files?.[0]); event.target.value = ""; }}
        />
      </main>
      {dragging && <div className="global-drop-overlay"><div><Upload size={30} /><strong>Drop course files to add them</strong><span>PDF, DOCX, or TXT · Stored locally</span></div></div>}
    </div>
  );
}

function PracticeQuestionsPage({
  datasets, uploads, selectedDatasetId, selectedUploadIds, lessonUploads,
  lessonUploadBusy, lessonProgress, useExamStyle, onChangeExamStyle, lessonUploadInput,
  onUploadLessons, onRemoveLesson, form, onOpenSettings,
  result, preferences, answers, busy, saved, onGenerate, onSave, onRevealAnswers,
  onExportCsv, onExportPdf, onBack, error, notice, onRateQuestion, onResetPreferences
}) {
  const [lessonDropActive, setLessonDropActive] = useState(false);
  const [questionSort, setQuestionSort] = useState("likelihood");
  const selectedDatasetName = datasets.find((dataset) => dataset.id === selectedDatasetId)?.name || "Built-in examples";
  const selectedUploadNames = uploads
    .filter((upload) => selectedUploadIds.includes(upload.id))
    .map((upload) => upload.name);
  const readyLessons = lessonUploads.filter((lesson) => lesson.status === "Ready");
  const examFiles = uploads.filter((upload) => (
    /\.pdf$/i.test(upload.name) && /(?:^|[_\s-])exam(?:[_\s.-]|$)/i.test(upload.name)
  ));
  const effectiveUploadNames = lessonUploads.length
    ? readyLessons.map((lesson) => lesson.name)
    : selectedUploadNames;
  const sortedQuestions = result?.questions ? [...result.questions].sort((left, right) => {
    if (questionSort === "topic") {
      return left.topic.localeCompare(right.topic) || left.lesson_order - right.lesson_order;
    }
    if (questionSort === "lesson") return left.lesson_order - right.lesson_order;
    return right.likelihood_score - left.likelihood_score
      || left.lesson_order - right.lesson_order;
  }) : [];
  const rejectionLabels = {
    already_a_question: "Already phrased as a question",
    below_topic_relevance: "Below the 45% topic-relevance threshold",
    candidate_validation_failed: "Failed source/answer validation",
    different_difficulty: "Did not match the selected difficulty",
    no_readable_words: "No readable words after text cleaning",
    no_safe_negation: "No safe true/false negation could be formed",
    not_enough_distinct_distractors: "Not enough distinct source-defined answer choices",
    distractor_is_answer_in_set: "A distractor conflicted with another selected answer",
    not_definition: "Not written in a supported definition format",
    outside_sentence_length: "Too short or long to use as a sentence",
    per_topic_limit_reached: "Skipped after reaching the per-topic limit",
    too_similar: "Too similar to another question",
    topic_not_selected: "Classified outside the selected topic"
  };
  return <section className="question-page">
    <div className="training-page-heading">
      <div>
        <div className="eyebrow"><span className="eyebrow-dot" /> SOURCE-GROUNDED STUDY TOOLS</div>
        <h1>Generate practice questions</h1>
        <p>Review your generation setup and create a source-grounded practice set. No external knowledge is used.</p>
      </div>
      <button className="button-outline question-back" onClick={onBack}>Back to Results</button>
    </div>

    <div className="question-priority-notice">
      <ShieldCheck size={17} />
      <span><strong>Likelihood is an estimate from your materials, not a guarantee of actual exam content.</strong> Past-exam signals are used only when matching exam PDFs are available locally.</span>
    </div>
    <div className="question-overview-card">
      <div className="section-heading">
        <div><h2><Sparkles size={17} /> Current setup</h2><p>Change question settings in Settings. Your choices are saved locally.</p></div>
        <button className="button-outline" type="button" onClick={onOpenSettings}>Question settings</button>
      </div>
      {readyLessons.length > 0 && <div className="question-overview-sources question-generating-from">
        <strong>Generating from: {readyLessons.length} lesson{readyLessons.length === 1 ? "" : "s"}</strong>
        <span title={readyLessons.map((lesson) => lesson.name).join(", ")}>
          {readyLessons.map((lesson) => lesson.name).join(", ")}
        </span>
      </div>}
      <div className="question-overview-sources question-style-reference">
        <strong>Style reference: {useExamStyle ? `${examFiles.length} exam file${examFiles.length === 1 ? "" : "s"}` : "Off"}</strong>
        <span>Exam files can provide question-stem patterns only; they never provide lesson facts or answer choices.</span>
      </div>
      <div className="question-overview-grid">
        <div><span>Training dataset</span><strong>{selectedDatasetName}</strong></div>
        <div><span>Source files</span><strong>{effectiveUploadNames.length} selected</strong></div>
        <div><span>Topic</span><strong>{form.topic}</strong></div>
        <div><span>Question format</span><strong>{form.question_type} · {form.difficulty}</strong></div>
        <div><span>Question count</span><strong>{form.number_of_questions} requested · {form.questions_per_topic} per topic max</strong></div>
      </div>
      <div className="question-overview-sources">
        <strong>Selected materials</strong>
        {effectiveUploadNames.length
          ? <span title={effectiveUploadNames.join(", ")}>{effectiveUploadNames.join(", ")}</span>
          : <span>No source files selected. Choose materials in Settings.</span>}
      </div>
      <div
        className={`lesson-upload-card ${lessonDropActive ? "lesson-upload-card-active" : ""}`}
        onDragOver={(event) => { event.preventDefault(); event.stopPropagation(); setLessonDropActive(true); }}
        onDragLeave={(event) => {
          event.stopPropagation();
          if (event.currentTarget === event.target) setLessonDropActive(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setLessonDropActive(false);
          onUploadLessons(event.dataTransfer.files);
        }}
      >
        <div className="lesson-upload-heading">
          <div><strong>Upload lessons</strong><span>Each listed lesson is a content source; all listed lessons generate together.</span></div>
          <button className="button-outline" type="button" disabled={lessonUploadBusy || busy} onClick={() => lessonUploadInput.current?.click()}>
            <Upload size={14} /> Choose lessons
          </button>
        </div>
        <button className="lesson-upload-drop" type="button" disabled={lessonUploadBusy || busy} onClick={() => lessonUploadInput.current?.click()}>
          {lessonUploadBusy
            ? <><span className="spinner" /> Uploading and reading lesson files locally...</>
            : <>Drop multiple PDF, DOCX, or TXT lessons here, or choose files</>}
        </button>
        {lessonUploads.length > 0 && <ul className="lesson-upload-list">
          {lessonUploads.map((lesson) => <li className={`lesson-upload-chip lesson-upload-chip-${lesson.status.toLowerCase()}`} key={lesson.id}>
            <FileText size={14} />
            <span className="lesson-upload-chip-copy">
              <strong title={lesson.name}>{lesson.name}</strong>
              <small>{lesson.status}{lesson.bytes ? ` · ${(lesson.bytes / (1024 * 1024)).toFixed(2)} MB` : ""}{lesson.error ? ` · ${lesson.error}` : ""}</small>
            </span>
            <button
              className="icon-button"
              type="button"
              onClick={() => onRemoveLesson(lesson.id)}
              disabled={lessonUploadBusy || busy}
              aria-label={`Remove ${lesson.name} from lesson generation`}
              title={`Remove ${lesson.name}`}
            ><X size={14} /></button>
          </li>)}
        </ul>}
        {lessonProgress && <div
          className={`lesson-processing lesson-processing-${lessonProgress.phase}`}
          role="status"
          aria-live="polite"
        >
          {lessonProgress.phase === "done"
            ? <Check size={14} />
            : lessonProgress.phase === "error"
              ? <AlertCircle size={14} />
              : <span className="spinner" />}
          <span>{lessonProgress.message}</span>
        </div>}
        <label className="lesson-style-toggle">
          <input type="checkbox" checked={useExamStyle} onChange={(event) => onChangeExamStyle(event.target.checked)} disabled={lessonUploadBusy || busy} />
          <span><strong>Use exam files for style</strong><small>Reuses common stem patterns and frequency; this is pattern reuse, not generative learning.</small></span>
        </label>
        <small className="lesson-upload-hint">PDF, DOCX, TXT · maximum 20 MB · scanned PDFs need local Tesseract OCR</small>
      </div>
      <button className="button-dark question-generate" disabled={busy || lessonUploadBusy || !effectiveUploadNames.length} onClick={onGenerate}>
        {busy ? <><span className="spinner" /> Generating...</> : <><Sparkles size={16} /> Generate Questions</>}
      </button>
      {!effectiveUploadNames.length && <p className="question-validation-note">Select materials in Settings or upload a lesson above.</p>}
    </div>

    {error && <div className="app-message app-message-error"><AlertCircle size={16} />{error}</div>}
    {notice && <div className="app-message app-message-success"><Check size={16} />{notice}</div>}

    {result && <div className="question-results question-print-area">
      {result.likely_topics && <section className="likely-topics-panel">
        <div className="section-heading"><div><h2>Likely exam topics</h2>
          <p>{result.exam_evidence_available
            ? `Ranked with ${result.exam_sources.length} readable past-exam file(s) and lesson emphasis.`
            : "Ranked by lesson emphasis only; no readable past-exam PDFs were found."}</p></div>
        </div>
        {result.classifier_training && <p className="classifier-training-note">
          Topic classifier fitted on demand during generation; new fit time {result.classifier_training.fit_seconds.toFixed(3)}s
          {result.classifier_training.cache_hits ? ` · ${result.classifier_training.cache_hits} cached reuse(s)` : ""}
          {result.classifier_training.exam_passages_added
            ? ` · ${result.classifier_training.exam_passages_added} confidently topic-labeled exam passages added`
            : " · no exam passages met the labeling confidence threshold"}.
        </p>}
        {result.style_reference_enabled && <p className="classifier-training-note">
          Reused {result.style_template_count} generic question-stem pattern(s) from {result.style_reference_files} exam file(s); no exam facts or answers were copied.
        </p>}
        {result.skipped_exam_files?.length > 0 && <p className="lesson-exam-note">
          Past-exam evidence could not be extracted from: {result.skipped_exam_files.join(", ")}
        </p>}
        <div className="likely-topic-list">{result.likely_topics.map((item) => <article className="likely-topic-card" key={item.topic}>
          <div><strong>{item.topic}</strong><span className={`likelihood-chip likelihood-${item.likelihood.toLowerCase()}`}>{item.likelihood} likelihood</span></div>
          <p>{item.why.join(" · ")}</p>
        </article>)}</div>
        <details className="lesson-passage-analysis">
          <summary>Lesson passage topic analysis · {result.lesson_analysis.passages.length} passages, {result.lesson_analysis.unclassified_count} unclassified</summary>
          <ul>{result.lesson_analysis.passages.map((passage, index) => <li key={`${index}-${passage.text.slice(0, 20)}`}>
            <div className="lesson-passage-copy">
              <span className="lesson-passage-topic">{passage.topic}</span>
              <span>{passage.text}</span>
              <span className="lesson-passage-signals">
                {passage.signals.heading && <span>Heading</span>}
                {passage.signals.definition && <span>Definition</span>}
                {passage.signals.list && <span>List item</span>}
                {passage.signals.emphasized_terms.map((term) => <span key={`em-${term}`}>Emphasis: {term}</span>)}
                {passage.signals.repeated_terms.map((term) => <span key={`rep-${term}`}>Repeated: {term}</span>)}
              </span>
            </div>
          </li>)}</ul>
        </details>
      </section>}
      <div className="section-heading"><div><h2>Practice set</h2><p>{result.lesson_analysis
        ? `Generated ${result.questions.length} of ${result.requested_count} requested. Skipped: ${result.diagnostics.skipped_low_confidence || 0} low-confidence, ${result.diagnostics.no_valid_distractors || 0} no valid distractors.`
        : `${result.questions.length} of ${result.requested_count} requested questions passed source and duplicate checks.`}</p></div></div>
      {result.shortfall > 0 && <div className="question-shortfall">{result.shortfall_reason} The app will not fill the set with unsupported questions.</div>}
      {result.diagnostics?.rejections?.length > 0 && <details className="question-diagnostics">
        <summary>Why some source text was skipped</summary>
        <p>Skipped {result.diagnostics.header_lines_skipped || 0} exam-header or student-information lines. Checked {result.diagnostics.source_sentence_count} text segments; {result.diagnostics.sentences_matching_filters} matched the basic text, difficulty, and question-format filters.</p>
        <ul>{result.diagnostics.rejections.map(({ reason, count }) => <li key={reason}>
          <span>{rejectionLabels[reason] || reason.replaceAll("_", " ")}</span><strong>{count}</strong>
        </li>)}</ul>
      </details>}
      <div className="question-actions no-print">
        <button className="button-outline" disabled={busy || !result.questions.length} onClick={onGenerate}><RotateCcw size={14} /> Generate Again</button>
        <button className="button-outline" disabled={busy || saved || !result.questions.length} onClick={onSave}><Save size={14} /> {saved ? "Saved Locally" : "Save Questions"}</button>
        <button className="button-outline" disabled={!result.questions.length} onClick={onExportCsv}><Download size={14} /> Export CSV</button>
        <button className="button-outline" disabled={!result.questions.length} onClick={onExportPdf}><Printer size={14} /> Export PDF</button>
        <button className="button-outline" disabled={busy || !result.questions.length} onClick={onRevealAnswers}><Eye size={14} /> {Object.keys(answers).length ? "Hide Answers" : "Show Answers"}</button>
      </div>
      {result.likely_topics && <label className="question-sort-control no-print">
        Sort questions
        <select value={questionSort} onChange={(event) => setQuestionSort(event.target.value)}>
          <option value="likelihood">Likelihood</option>
          <option value="topic">Topic</option>
          <option value="lesson">Lesson order</option>
        </select>
      </label>}
      <div className="question-cards">
        {(result.likely_topics ? sortedQuestions : result.questions).map((item, index) => <article className="practice-question-card" key={item.id}>
          <div className="practice-question-meta">
            <span>QUESTION {String(index + 1).padStart(2, "0")}</span>
            <span>{item.topic}</span>
            <span>{item.difficulty}</span>
            {item.likelihood && <span className={`likelihood-chip likelihood-${item.likelihood.toLowerCase()}`}>{item.likelihood} likelihood</span>}
            {!item.likelihood && <><span>PRIORITY {Math.round(item.priority_score * 100)}%</span><span>RELEVANCE {Math.round(item.confidence_score * 100)}%</span></>}
          </div>
          <h3>{item.question_text}</h3>
          {item.choices.length > 0 && <ol className="practice-choices">
            {item.choices.map((choice, choiceIndex) => <li key={`${item.id}-${choiceIndex}`}><span>{String.fromCharCode(65 + choiceIndex)}.</span>{choice}</li>)}
          </ol>}
          {answers[item.id] && <div className="practice-answer"><strong>Answer</strong><span>{answers[item.id]}</span></div>}
          {item.likelihood_why && <p className="question-likelihood-why"><strong>Why:</strong> {item.likelihood_why.join(" · ")}</p>}
          <QuestionFeedbackButtons
            item={item}
            onRate={(rating) => onRateQuestion(item, rating, result.generation_id)}
          />
          <details className="practice-source">
            <summary>View source reference</summary>
            <p>{item.source_reference}</p>
          </details>
        </article>)}
      </div>
      {!result.lesson_analysis && result.topic_priorities?.length > 0 && <details className="question-priority-breakdown">
        <summary>Transparent topic priority breakdown</summary>
        <div className="priority-table-wrap"><table><thead><tr><th>Topic</th><th>Priority</th><th>Frequency</th><th>Material</th><th>Model</th><th>Recent*</th></tr></thead>
          <tbody>{result.topic_priorities.map((item) => <tr key={item.topic}>
            <td>{item.topic}</td><td>{Math.round(item.priority_score * 100)}%</td>
            <td>{Math.round(item.frequency_score * 100)}%</td><td>{Math.round(item.course_importance_score * 100)}%</td>
            <td>{Math.round(item.model_confidence_score * 100)}%</td><td>{Math.round(item.recent_occurrence_score * 100)}%</td>
          </tr>)}</tbody></table></div>
        <p>*Recent occurrence cannot be established because the selected training records do not contain per-example dates; the score is neutral, not an inferred chronology.</p>
      </details>}
      <p className="output-disclaimer">{result.disclaimer}</p>
    </div>}
  </section>;
}

function SavedQuestionsPage({
  sessions, busy, expandedSessions, setExpandedSessions, visibleAnswerSessions,
  onToggleAnswers, onToggleAllAnswers, onRename, onDelete, onRateQuestion, error
}) {
  const questionSessions = sessions.filter((session) => session.questions.length);
  const allAnswersVisible = questionSessions.length > 0
    && questionSessions.every((session) => visibleAnswerSessions.includes(session.id));
  const allExpanded = sessions.length > 0
    && sessions.every((session) => expandedSessions.includes(session.id));

  return <section className="question-page">
    <div className="training-page-heading">
      <div>
        <div className="eyebrow"><span className="eyebrow-dot" /> SAVED LOCALLY</div>
        <h1>Saved questions</h1>
        <p>Questions are grouped by practice-question generation session. Answers stay hidden until revealed.</p>
      </div>
      {sessions.length > 0 && <div className="saved-session-global-actions">
        <button className="button-outline" onClick={() => setExpandedSessions(
          allExpanded ? [] : sessions.map((session) => session.id)
        )}>
          {allExpanded ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
          {allExpanded ? "Collapse All" : "Expand All"}
        </button>
        {questionSessions.length > 0 && <button
          className="button-outline"
          disabled={busy}
          onClick={onToggleAllAnswers}
        >
          <Eye size={14} /> {busy ? "Loading..." : allAnswersVisible ? "Hide All Answers" : "Show All Answers"}
        </button>}
      </div>}
    </div>

    {error && <div className="app-message app-message-error"><AlertCircle size={16} />{error}</div>}
    {busy && !sessions.length
      ? <div className="output-empty"><span className="spinner" /><strong>Loading saved questions...</strong></div>
      : sessions.length
        ? <div className="saved-session-list">
          {sessions.map((session) => {
            const expanded = expandedSessions.includes(session.id);
            const answersVisible = visibleAnswerSessions.includes(session.id);
            const createdAt = new Date(session.created_at);
            const formattedDate = Number.isNaN(createdAt.getTime())
              ? session.created_at : createdAt.toLocaleString();
            return <article className="saved-session-card" key={session.id}>
              <div className="saved-session-heading">
                <button
                  className="saved-session-expand"
                  aria-expanded={expanded}
                  onClick={() => setExpandedSessions((current) => expanded
                    ? current.filter((id) => id !== session.id)
                    : [...current, session.id]
                  )}
                >
                  {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  <span className="saved-session-heading-copy">
                    <strong>{session.title}</strong>
                    <small>{formattedDate} · {session.question_count} question{session.question_count === 1 ? "" : "s"}</small>
                  </span>
                </button>
                <div className="saved-session-actions">
                  <button className="button-outline" onClick={() => onRename(session)} aria-label={`Rename ${session.title}`} title="Rename session">
                    <Pencil size={13} />
                  </button>
                  <button className="button-outline saved-session-delete" onClick={() => onDelete(session)} aria-label={`Delete ${session.title}`} title="Delete session">
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
              <div className="saved-session-tags">
                <span>{session.topic}</span>
                <span>{session.source_file}</span>
                <span>{session.difficulty}</span>
                <span>{session.question_type}</span>
              </div>
              {expanded && <>
                <div className="saved-session-answer-actions">
                  <button
                    className="button-outline"
                    disabled={busy}
                    onClick={() => onToggleAnswers(session.id)}
                  >
                    <Eye size={13} /> {busy ? "Loading..." : answersVisible ? "Hide Answers" : "Show Answers"}
                  </button>
                </div>
                <div className="question-cards">
                  {session.questions.map((item, index) => <article className="practice-question-card" key={item.id}>
                    <div className="practice-question-meta">
                      <span>QUESTION {String(index + 1).padStart(2, "0")}</span>
                      <span>{item.topic}</span>
                      <span>{item.difficulty}</span>
                      <span>{item.question_type}</span>
                    </div>
                    <h3>{item.question_text}</h3>
                    {item.choices?.length > 0 && <ol className="practice-choices">
                      {item.choices.map((choice, choiceIndex) => (
                        <li key={`${item.id}-${choiceIndex}`}>
                          <span>{String.fromCharCode(65 + choiceIndex)}.</span>{choice}
                        </li>
                      ))}
                    </ol>}
                    {answersVisible && item.correct_answer && <div className="practice-answer">
                      <strong>Answer</strong><span>{item.correct_answer}</span>
                    </div>}
                    <QuestionFeedbackButtons
                      item={item}
                      onRate={(rating) => onRateQuestion(item, rating)}
                    />
                    <details className="practice-source">
                      <summary>View source reference</summary>
                      <p>{item.source_reference}</p>
                      <small>{item.source_dataset_name} · {item.source_material_name}</small>
                    </details>
                  </article>)}
                </div>
              </>}
            </article>;
          })}
        </div>
        : <div className="output-empty">
          <NotebookText size={24} />
          <strong>No saved questions yet</strong>
          <span>Generate a practice set and choose “Save Questions” to keep it here.</span>
        </div>}
  </section>;
}

function QuestionPreferenceNotice({ preferences, onReset }) {
  const liked = preferences.learned_preferences?.slice(0, 3).map((item) => {
    if (item.feature === "wording_pattern") return `${item.value} wording`;
    return item.value;
  }) || [];
  return <div className="question-feedback-notice">
    <ThumbsUp size={16} />
    <span>
      <strong>Using your feedback ({preferences.ratings_recorded} rating{preferences.ratings_recorded === 1 ? "" : "s"})</strong>
      {liked.length ? ` · Current positive signals: ${liked.join(", ")}.` : " · Ratings guide future question ordering by format, wording, and length."}
    </span>
    {preferences.ratings_recorded > 0 && <button className="button-outline" onClick={onReset}>Reset preferences</button>}
  </div>;
}

function QuestionFeedbackButtons({ item, onRate }) {
  return <div className="question-feedback-actions no-print" aria-label="Rate this question">
    <span>Was this a good question?</span>
    <button
      className={`button-outline ${item.feedback_rating === 1 ? "question-feedback-selected" : ""}`}
      type="button"
      aria-pressed={item.feedback_rating === 1}
      onClick={() => onRate(1)}
    >
      <ThumbsUp size={13} /> Good
    </button>
    <button
      className={`button-outline ${item.feedback_rating === -1 ? "question-feedback-selected" : ""}`}
      type="button"
      aria-pressed={item.feedback_rating === -1}
      onClick={() => onRate(-1)}
    >
      <ThumbsDown size={13} /> Not for me
    </button>
  </div>;
}

function UploadPage({
  uploads, datasets, selectedDatasetId, setSelectedDatasetId, trainingBusy, queuedFiles,
  addFiles, fileInput, removeQueued, includeSavedFile, removeUpload, analyze,
  busy, progress, latest, error, notice, dragging
}) {
  return (
    <section className="upload-page">
      <div className="upload-page-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-dot" /> PRIVATE STUDY WORKSPACE</div>
          <h1>Upload course materials</h1>
          <p>Add notes, quizzes, or past exams. The app extracts the text and presents topic predictions here.</p>
        </div>
        <div className="local-badge"><ShieldCheck size={15} /> Local-only storage</div>
      </div>

      <div className={`upload-layout ${dragging ? "upload-layout-dragging" : ""}`}>
        <div className="upload-controls">
          <button className="upload-dropzone" onClick={() => fileInput.current?.click()}>
            <span className="upload-drop-icon"><FileUp size={25} /></span>
            <strong>Choose files or drag them here</strong>
            <span>PDF, DOCX, and TXT files · Stored on this device</span>
            <span className="button-dark upload-browse"><FolderOpen size={15} /> Browse files</span>
          </button>

          {queuedFiles.length > 0 && <div className="queued-files">
            <div className="section-heading"><div><h3>Ready to analyze</h3><p>{queuedFiles.length} file{queuedFiles.length === 1 ? "" : "s"} selected</p></div></div>
            {queuedFiles.map((item) => <div className="queued-file" key={item.id}>
              <FileText size={17} />
              <span><strong>{item.name}</strong><small>{formatBytes(item.bytes)}{item.saved ? " · saved locally" : ""}</small></span>
              <button className="icon-button" onClick={() => removeQueued(item.id)} aria-label={`Remove ${item.name}`}><X size={16} /></button>
            </div>)}
          </div>}

          <div className="analysis-options">
            <label className="dataset-choice"><span><strong>Training data for this prediction</strong><small>Models are trained locally and cached for reuse when practical.</small></span>
              <select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)} disabled={busy || trainingBusy}>
                {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>
                  {dataset.name} · {dataset.example_count} examples · {dataset.topic_count} topics
                </option>)}
              </select>
            </label>
            <div className="ensemble-note"><BrainCircuit size={15} /><span><strong>Combined prediction</strong><small>TF-IDF features are analyzed by SVM and Naive Bayes together.</small></span></div>
            <button className="button-dark analyze-button" disabled={busy || trainingBusy || !queuedFiles.length} onClick={analyze}>
              {busy ? <><span className="spinner" /> Analyzing materials...</> : <><BrainCircuit size={16} /> Analyze materials</>}
            </button>
          </div>
          <p className="storage-note"><ShieldCheck size={14} /> Files and predictions are saved locally on this computer.</p>
        </div>

        <aside className="saved-materials">
          <div className="section-heading"><div><h3>Saved materials</h3><p>Files stored locally for later use</p></div><span className="count-badge">{uploads.length}</span></div>
          {uploads.length ? <div className="saved-file-list">
            {uploads.map((upload) => <div className="saved-file" key={upload.id}>
              <span className="saved-file-icon"><FileText size={16} /></span>
              <span className="saved-file-copy"><strong title={upload.name}>{upload.name}</strong><small>{formatBytes(upload.bytes)} · {formatDate(upload.uploaded_at)}</small></span>
              <button className="icon-button saved-use" title={`Use ${upload.name} in analysis`} aria-label={`Use ${upload.name} in analysis`} onClick={() => includeSavedFile(upload)}><RotateCcw size={15} /></button>
              <button className="icon-button saved-delete" title={`Delete ${upload.name}`} aria-label={`Delete ${upload.name}`} onClick={() => removeUpload(upload)}><Trash2 size={15} /></button>
            </div>)}
          </div> : <div className="saved-empty"><FolderOpen size={22} /><span>Uploaded files will appear here.</span></div>}
        </aside>
      </div>

      {(busy || progress.length > 0) && <ProgressPanel progress={progress} busy={busy} />}
      {error && <div className="app-message app-message-error"><AlertCircle size={16} />{error}</div>}
      {notice && !error && <div className="app-message app-message-success"><Check size={16} />{notice}</div>}
      <AnalysisOutput analysis={latest} />
    </section>
  );
}

function ProgressPanel({ progress, busy }) {
  const phases = progress.length ? progress : [
    { phase: "extraction", label: "Reading course materials", state: "active" },
    { phase: "cleaning", label: "Cleaning and preparing text", state: "pending" },
    { phase: "features", label: "Extracting text features", state: "pending" },
    { phase: "analysis", label: "Ranking likely topics", state: "pending" }
  ];
  return <div className="progress-panel">
    <div className="section-heading"><div><h3>{busy ? "Analyzing materials" : "Analysis steps"}</h3><p>Text stays on this device.</p></div><span className="progress-live">{busy ? "IN PROGRESS" : "COMPLETE"}</span></div>
    <div className="progress-steps">{phases.map((item) => <div className={`progress-step progress-step-${item.state}`} key={item.phase}>
      <span>{item.state === "done" ? <Check size={12} /> : item.state === "active" && busy ? <span className="spinner-small" /> : null}</span>{item.label}
    </div>)}</div>
  </div>;
}

function AnalysisOutput({ analysis }) {
  return <section className="analysis-output">
    <div className="section-heading"><div><h2>Topic predictions</h2><p>Most likely topics based on the analyzed material</p></div>
      {analysis && <span className="output-method">{analysis.method || "TF-IDF + SVM + Naive Bayes ensemble"}</span>}
    </div>
    {!analysis ? <div className="output-empty"><BrainCircuit size={24} /><strong>Your output will appear here</strong><span>Upload materials and run an analysis to see ranked topics.</span></div>
      : <div className="output-card">
        <div className="output-card-heading"><div><span className="output-source-icon"><FolderOpen size={17} /></span><span><strong>{analysis.label || "Uploaded materials"}</strong><small>Saved locally · {formatDate(analysis.created_at)}</small></span></div><span className="saved-indicator"><span /> SAVED</span></div>
        <div className="prediction-list">{analysis.predictions.map((prediction, index) => <div className="prediction-row" key={`${prediction.topic}-${index}`}>
          <span className="prediction-rank">{String(index + 1).padStart(2, "0")}</span>
          <span className="prediction-topic"><strong>{prediction.topic}</strong><span><i style={{ width: `${Math.max(0, Math.min(100, prediction.confidence * 100))}%` }} /></span></span>
          <strong className="prediction-confidence">{Math.round(prediction.confidence * 100)}%</strong>
        </div>)}</div>
        <p className="output-disclaimer">Training data: {analysis.training_dataset_name || "Built-in examples"}. SVM and Naive Bayes scores are averaged equally over shared TF-IDF features.</p>
      </div>}
  </section>;
}

function TrainingDataPage({
  uploads, datasets, selectedDatasetId, datasetName, setDatasetName, selectedUploads,
  setSelectedUploads, labels, setLabels, busy, fileInput, csvInput,
  onImportCsv, onCreateDocuments, onRemoveUpload, onDeleteDataset, error, notice
}) {
  const [evaluation, setEvaluation] = useState(null);
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const toggleUpload = (id) => setSelectedUploads((current) => current.includes(id)
    ? current.filter((item) => item !== id)
    : [...current, id]);
  const evaluateSelectedDataset = async () => {
    setEvaluationBusy(true);
    setEvaluationError("");
    setEvaluation(null);
    try {
      setEvaluation(await api.evaluateTrainingDataset(selectedDatasetId));
    } catch (requestError) {
      setEvaluationError(`Evaluation failed: ${requestError.message}`);
    } finally {
      setEvaluationBusy(false);
    }
  };

  return <section className="training-page">
    <div className="training-page-heading">
      <div>
        <div className="eyebrow"><span className="eyebrow-dot" /> LOCAL MODEL TRAINING</div>
        <h1>Manage training data</h1>
        <p>Import labeled examples or assign topics to your documents. Select a saved dataset on the Upload tab when you want to use it for a prediction.</p>
      </div>
      <div className="local-badge"><ShieldCheck size={15} /> Local-only storage</div>
    </div>
    <div className="training-data-panel model-evaluation-panel">
      <div className="section-heading">
        <div><h2><BrainCircuit size={16} /> Model evaluation</h2>
          <p>Evaluate {datasets.find((dataset) => dataset.id === selectedDatasetId)?.name || "the selected dataset"} with stratified cross-validation.</p>
        </div>
        <button className="button-outline" onClick={evaluateSelectedDataset} disabled={evaluationBusy || busy}>
          {evaluationBusy ? <span className="spinner" /> : <Check size={14} />}
          {evaluationBusy ? "Evaluating…" : "Evaluate selected dataset"}
        </button>
      </div>
      <p className="evaluation-disclaimer">Metrics summarize predictions on held-out folds of the labeled dataset. They do not measure performance on future exams.</p>
      {evaluationError && <div className="app-message app-message-error"><AlertCircle size={16} />{evaluationError}</div>}
      {evaluation && <div className="evaluation-report">
        <div className="evaluation-report-meta">
          <strong>{evaluation.training_dataset_name}</strong>
          <span>{evaluation.examples_evaluated} examples · {evaluation.folds} folds · {evaluation.examples_excluded} unreadable rows excluded</span>
        </div>
        <div className="evaluation-metrics">
          {[
            ["Accuracy", evaluation.metrics.accuracy],
            ["Macro precision", evaluation.metrics.macro_precision],
            ["Macro recall", evaluation.metrics.macro_recall],
            ["Macro F1", evaluation.metrics.macro_f1]
          ].map(([label, score]) => <div key={label}>
            <span>{label}</span><strong>{(score * 100).toFixed(1)}%</strong>
          </div>)}
        </div>
        <div className="evaluation-matrix-wrap">
          <strong>Confusion matrix <span>(rows: actual · columns: predicted)</span></strong>
          <table className="evaluation-matrix">
            <thead><tr><th>Actual / predicted</th>{evaluation.confusion_matrix.labels.map((label) => <th key={label}>{label}</th>)}</tr></thead>
            <tbody>{evaluation.confusion_matrix.values.map((row, rowIndex) => <tr key={evaluation.confusion_matrix.labels[rowIndex]}>
              <th>{evaluation.confusion_matrix.labels[rowIndex]}</th>
              {row.map((count, columnIndex) => <td key={evaluation.confusion_matrix.labels[columnIndex]}>{count}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
      </div>}
    </div>
    <div className="training-data-panel">
    <div className="section-heading">
      <div><h2><Database size={16} /> Create a dataset</h2><p>CSV rows or labeled documents become training examples for SVM and Naive Bayes.</p></div>
      <span className="count-badge">{datasets.length}</span>
    </div>
    <div className="training-data-grid">
      <div className="training-builder">
        <label className="training-name-field"><span>Dataset name</span>
          <input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} maxLength={100} placeholder="e.g. Biology midterm" />
        </label>
        <div className="training-actions">
          <button className="button-outline" onClick={() => csvInput.current?.click()} disabled={busy}>
            {busy ? <span className="spinner" /> : <FileSpreadsheet size={15} />} Import labeled CSV
          </button>
          <button className="button-outline" onClick={() => fileInput.current?.click()} disabled={busy}>
            <FileUp size={15} /> Add documents
          </button>
        </div>
        <p className="training-help">CSV format: one header row with <code>topic,text</code> columns. Uploaded documents are saved locally, but do not become an available dataset until you label them and choose <strong>Create from labeled documents</strong>.</p>
        {uploads.length ? <div className="training-upload-list">
          {uploads.map((upload) => {
            const selected = selectedUploads.includes(upload.id);
            return <div className={`training-upload-row ${selected ? "training-upload-selected" : ""}`} key={upload.id}>
              <label className="training-upload-select">
                <input type="checkbox" checked={selected} onChange={() => toggleUpload(upload.id)} />
                <span title={upload.name}>{upload.name}</span>
              </label>
              <div className="training-upload-actions">
                <input
                  className="training-topic-input"
                  value={labels[upload.id] || ""}
                  onChange={(event) => setLabels((current) => ({ ...current, [upload.id]: event.target.value }))}
                  aria-label={`Topic label for ${upload.name}`}
                  placeholder="Topic label"
                  disabled={!selected}
                />
                <button
                  className="icon-button training-upload-delete"
                  type="button"
                  onClick={() => onRemoveUpload(upload)}
                  disabled={busy}
                  title={`Delete ${upload.name} from this device`}
                  aria-label={`Delete uploaded document ${upload.name}`}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>;
          })}
          <button className="button-dark create-dataset-button" onClick={onCreateDocuments} disabled={busy || selectedUploads.length < 2 || !datasetName.trim()}>
            <Database size={14} /> Create from labeled documents
          </button>
        </div> : <div className="training-empty">Add at least two documents and label each with its topic to train from documents.</div>}
      </div>
      <div className="training-saved">
        <h3>Available datasets</h3>
        <div className="dataset-list">
          {datasets.map((dataset) => <div className="dataset-list-row" key={dataset.id}>
            <span className="dataset-list-icon"><Database size={15} /></span>
            <span className="dataset-list-copy"><strong>{dataset.name}</strong><small>{dataset.example_count} examples · {dataset.topic_count} topics · {dataset.source_type}</small></span>
            {dataset.id !== "builtin" && <button className="icon-button saved-delete" onClick={() => onDeleteDataset(dataset)} title={`Delete ${dataset.name}`} aria-label={`Delete dataset ${dataset.name}`}><Trash2 size={15} /></button>}
          </div>)}
        </div>
      </div>
    </div>
    </div>
    {(preferences?.ratings_recorded > 0 || result?.preference_learning?.applied) && <QuestionPreferenceNotice
    preferences={preferences}
    onReset={onResetPreferences}
    />}

    {error && <div className="app-message app-message-error"><AlertCircle size={16} />{error}</div>}
    {notice && !error && <div className="app-message app-message-success"><Check size={16} />{notice}</div>}
  </section>;
}

function SettingsPage({
  settings, setSettings, datasets, uploads, selectedDatasetId,
  setSelectedDatasetId, selectedUploadIds, setSelectedUploadIds, topics,
  questionForm, setQuestionForm, saveSettings, busy, appVersion,
  onCheckForUpdates, updateCheckBusy, error, notice
}) {
  const setQuestionField = (key, value) => setQuestionForm((current) => ({
    ...current,
    [key]: value
  }));
  const weightTotal = Object.values(questionForm.weights)
    .reduce((sum, value) => sum + Number(value || 0), 0);
  const weightLabels = {
    historical_frequency: "Selected-dataset topic frequency",
    course_importance: "Course-material importance",
    model_confidence: "ML topic confidence",
    recent_occurrence: "Recent occurrence"
  };
  const toggleSource = (id) => setSelectedUploadIds((current) => current.includes(id)
    ? current.filter((item) => item !== id)
    : [...current, id]);

  return <section className="simple-page settings-page">
    <div className="eyebrow">CLINTHINKER© WORKSPACE</div><h1>Settings</h1>
    <p>Choose interface preferences and configure how practice questions are generated. Settings are saved locally.</p>
    <div className="simple-settings-card">
      <label><span><strong>Appearance</strong><small>Choose your workspace color theme.</small></span>
        <select value={settings.appearance} onChange={(event) => setSettings((current) => ({ ...current, appearance: event.target.value }))}>
          <option value="orange">Black &amp; Orange</option>
          <option value="pink">Clinthinker White &amp; Pink</option>
          <option value="regal">Alamo&apos;s Regal Nature (Gold &amp; Purple)</option>
        </select>
      </label>
      <label><span><strong>Language</strong><small>Language used by the interface.</small></span>
        <select value={settings.language} onChange={(event) => setSettings((current) => ({ ...current, language: event.target.value }))}><option>English</option></select>
      </label>
      <div className="app-version-row">
        <span><strong>Version {appVersion || "..."}</strong></span>
        <button className="button-outline" onClick={onCheckForUpdates} disabled={updateCheckBusy}>
          {updateCheckBusy ? "Checking..." : "Check for updates"}
        </button>
      </div>
    </div>
    <div className="settings-question-generation">
      <div className="section-heading">
        <div><h2><Sparkles size={17} /> Question generation</h2>
          <p>Choose the training dataset, source files, and question format used on the Generate Practice Questions overview.</p>
        </div>
      </div>
      <div className="question-controls-grid">
        <label className="question-control"><span>Training dataset for topic guidance</span>
          <select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)}>
            {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>
              {dataset.name} · {dataset.topic_count} topics
            </option>)}
          </select>
        </label>
        <label className="question-control"><span>Topic</span>
          <select value={questionForm.topic} onChange={(event) => setQuestionField("topic", event.target.value)}>
            <option>All Topics</option>
            {topics.map((topic) => <option key={topic}>{topic}</option>)}
          </select>
        </label>
        <label className="question-control"><span>Question type</span>
          <select value={questionForm.question_type} onChange={(event) => setQuestionField("question_type", event.target.value)}>
            {["Multiple Choice", "True or False", "Identification"].map((type) => <option key={type}>{type}</option>)}
          </select>
        </label>
        <label className="question-control"><span>Difficulty</span>
          <select value={questionForm.difficulty} onChange={(event) => setQuestionField("difficulty", event.target.value)}>
            {["Easy", "Moderate", "Difficult"].map((level) => <option key={level}>{level}</option>)}
          </select>
        </label>
        <label className="question-control"><span>Number of questions</span>
          <input type="number" min="1" max="50" value={questionForm.number_of_questions}
            onChange={(event) => setQuestionField("number_of_questions", event.target.value)} />
        </label>
        <label className="question-control"><span>Questions per topic</span>
          <input type="number" min="1" max="50" value={questionForm.questions_per_topic}
            onChange={(event) => setQuestionField("questions_per_topic", event.target.value)} />
        </label>
      </div>
      <p className="question-difficulty-note">Difficulty is estimated from sentence length and complexity. Sources also need at least 45% model relevance for the selected dataset.</p>

      <div className="question-source-picker">
        <div className="question-subheading">
          <strong>Course materials used as question sources</strong>
          <span>All saved files are used by default, including files added later. Uncheck any file to exclude it. Training examples guide classification; custom training rows can also be candidates in standard generation, but never in lesson mode.</span>
          {uploads.length > 0 && <div className="question-source-actions">
            <button className="button-outline" type="button" onClick={() => setSelectedUploadIds(uploads.map((upload) => upload.id))}>Select all</button>
            <button className="button-outline" type="button" onClick={() => setSelectedUploadIds([])}>Clear selection</button>
            <span>{selectedUploadIds.length} of {uploads.length} selected</span>
          </div>}
        </div>
        {uploads.length ? <div className="question-upload-list">
          {uploads.map((upload) => <label key={upload.id} className="question-upload-choice">
            <input type="checkbox" checked={selectedUploadIds.includes(upload.id)} onChange={() => toggleSource(upload.id)} />
            <FileText size={15} /><span>{upload.name}</span>
          </label>)}
        </div> : <div className="training-empty">Upload course materials on the Upload page first.</div>}
      </div>

      <details className="question-weights">
        <summary>Configure topic priority score <span>{weightTotal}% total</span></summary>
        <p>Weights must total 100%. Historical examples have no timestamps, so recency uses a neutral score.</p>
        <div className="question-weight-grid">{Object.entries(weightLabels).map(([key, label]) =>
          <label className="question-control" key={key}><span>{label} (%)</span>
            <input type="number" min="0" max="100" value={questionForm.weights[key]}
              onChange={(event) => setQuestionForm((current) => ({
                ...current, weights: { ...current.weights, [key]: event.target.value }
              }))} />
          </label>
        )}</div>
        <label className="question-control question-threshold"><span>Duplicate similarity threshold (%)</span>
          <input type="number" min="50" max="100" value={questionForm.similarity_threshold}
            onChange={(event) => setQuestionField("similarity_threshold", event.target.value)} />
        </label>
      </details>
      {weightTotal !== 100 && <p className="question-validation-note">Priority weights currently total {weightTotal}%. Adjust them to 100% before saving.</p>}
    </div>
    <button className="button-dark settings-save" onClick={saveSettings} disabled={busy || weightTotal !== 100}>
      {busy ? "Saving..." : "Save settings"}
    </button>
    {error && <div className="app-message app-message-error"><AlertCircle size={16} />{error}</div>}
    {notice && !error && <div className="app-message app-message-success"><Check size={16} />{notice}</div>}
    <div className="offline-note"><ShieldCheck size={17} /><span><strong>Offline by design</strong><small>Clinthinker© runs the prediction service and stores data on this computer. No cloud account or external data service is used.</small></span></div>
  </section>;
}

function HelpPage() {
  return <section className="simple-page">
    <div className="eyebrow">GETTING STARTED</div><h1>Help</h1>
    <p>Use the local assistant to turn study materials into a short list of likely exam topics.</p>
    <div className="help-steps">
      <div><span>01</span><strong>Add materials</strong><p>Choose PDF, DOCX, or TXT notes, quizzes, and past exams. Uploaded copies stay in local storage.</p></div>
      <div><span>02</span><strong>Choose training data</strong><p>Select built-in examples or import a labeled CSV with topic/text columns. You can also upload documents and assign a topic to each.</p></div>
      <div><span>03</span><strong>Analyze locally</strong><p>The SVM and Naive Bayes classifier is fitted on demand from the selected dataset and cached in memory. Lesson generation can add confidently topic-labeled exam passages. It classifies topics; it does not learn how to write questions.</p></div>
    </div>
    <div className="offline-note"><ShieldCheck size={17} /><span><strong>Local data folder</strong><small>Uploads and training datasets are stored in <code>%LOCALAPPDATA%\Clinthinker©\storage\</code>. Existing Topic Prediction Assistant data is moved here on first launch.</small></span></div>
  </section>;
}

function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "Recently";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}
