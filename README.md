# Clinthinker©

A local-only desktop app for analyzing course materials and showing ranked
topic predictions. The Windows app uses the supplied burger image as its
executable and PWA icon, and opens the existing responsive interface in a
native desktop window. Prediction, parsing, and persistent storage run locally
in Python; the React interface is bundled with the app.

Choose **Black & Orange**, **Clinthinker White & Pink**, or **Alamo's Regal
Nature (Gold & Purple)** under **Settings → Appearance**. The current
black-and-orange theme remains the default.

## Build the Windows app

Requirements:

- Windows 10 or 11
- 64-bit Python 3.10+ and 64-bit Node.js 20.19+ (or 22.12+) with npm
- Inno Setup 6, with `ISCC.exe` on `PATH` or installed in its default Program Files folder
- Internet access on the build computer to install dependencies and download
  Microsoft's WebView2 offline installer
- For OCR of scanned PDFs, install Tesseract OCR with English language data on
  the computer running the app and add `tesseract.exe` to `PATH`

From PowerShell in this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build-windows-app.ps1
```

The builder compiles the interface and packages the Python API, machine
learning and document-reading dependencies, and includes Microsoft's x64
WebView2 offline runtime installer in the app folder. It also creates a
per-user Inno Setup installer from the complete PyInstaller `--onedir` output,
including `Prerequisites`. The installer installs under
`%LOCALAPPDATA%\Programs\Clinthinker`, closes the running app when upgrading,
and does not modify the separate `%LOCALAPPDATA%\Clinthinker©\storage` data
folder. The outputs are:

```text
release\Clinthinker©\Clinthinker©.exe
release\Clinthinker-Setup-2.0.0.exe
```

Keep the entire generated app folder together when moving or sharing it,
including `Prerequisites\MicrosoftEdgeWebView2RuntimeInstallerX64.exe`.
Double-click the EXE to open it; no browser, Python, Node.js, or separate
backend terminal is needed on the target computer. If WebView2 is not already
installed, the app offers to install it for the current Windows user from the
included offline installer. The first build can take several minutes and
produces a sizeable app folder because Python, scikit-learn, document parsers,
and the WebView2 installer are included.

The setup installer is built by `installer.iss` after PyInstaller completes.
The build script reads `APP_VERSION` from `backend\app\main.py` and passes it
to Inno Setup for the plain-ASCII installer filename. Until a matching release
installer is published, `version.json` intentionally contains no
`installer_url`, and the app will not offer an update when the manifest version
matches its own.

## Run from source

Install the backend dependencies and start the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, build the desktop app and serve the interface:

```powershell
cd frontend
npm install
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

`npm run build` builds the web interface and refreshes the packaged desktop app
at `release\Clinthinker©\Clinthinker©.exe`. Use `npm run build:web` when only a
frontend bundle is needed. Open <http://127.0.0.1:4173> to preview the interface.
Both Vite development and preview servers proxy API requests to the local
backend.

## Use the app

1. Upload PDF, DOCX, or TXT course materials (up to 20 MB each) to analyze.
2. In **Training data for this prediction**, choose the built-in examples or
   one of your own saved datasets.
3. Open the **Training Data** tab in the menu. Import a labeled CSV with
   `topic` and `text` columns, or add PDF, DOCX, or TXT files, assign a topic
   to each, and create a dataset. Training datasets need at least two topic
   labels.
4. Select **Analyze materials**. Both SVM and Naive Bayes are fitted locally
   on demand from the selected dataset using shared TF-IDF features, and their
   normalized topic scores are averaged equally. Models are cached in memory
   for reuse; no fitted model is written to disk.
5. Delete your custom datasets from **Available datasets**. Deleting a dataset
   does not delete the original uploaded documents.

On **Training Data**, use **Evaluate selected dataset** to estimate classifier
performance with stratified cross-validation. The report includes accuracy,
macro-averaged precision, recall and F1, plus a confusion matrix. Each fold
fits its own TF-IDF vocabulary and classifier using only that fold's training
rows. At least two readable labeled examples per topic are required. These
metrics estimate performance on the selected labeled dataset; they do not
measure accuracy on future examinations or guarantee predicted exam topics.

Custom CSV rows and labeled document text are stored in the local SQLite
database. A normal prediction uses the one selected training dataset; it does
not silently combine your custom examples with built-in examples.

## Generate practice questions

Configure question generation under **Settings → Question generation**. Choose
the training dataset, topic, question type, difficulty, question count,
per-topic limit, and saved course files there, then select **Save settings**.
All saved course files are used by default, including files added later. Use
the individual checkboxes, **Select all**, or **Clear selection** to change
the sources; the all-files default stays current as files are added. The Generate Practice
Questions page is an overview of that setup and the resulting practice set.
The total selected, readable text is limited to 500,000 characters. The
built-in examples and each custom dataset are separate choices. Use **Training
Data** to import a labeled CSV or create another dataset from labeled
documents. A custom dataset is selected automatically after it is created or
imported. Saved settings are stored locally on this device.
Supported types are Multiple
Choice, True or False, and Identification. Questions are extracted and formed
from source sentences/definitions; multiple-choice distractors must also come
from source-defined terms. True/False sets alternate source-affirmed statements
with negated statements when the source sentence has a supported, clear
negation form, and skips sentence forms it cannot safely negate. Image-based PDF
pages with little extractable text can be processed locally with Tesseract OCR
(up to 100 pages per file); searchable text is preferred when available. Install
Tesseract OCR with English language data separately and make sure `tesseract.exe`
is on `PATH`. If OCR is needed but Tesseract is unavailable, the app explains
what to install rather than treating the scan as readable text.
Question generation skips common exam headings, page numbers, and student or
course metadata fields such as Name and Section; it uses the remaining readable
course content to build new questions.
The generator does not use an external language model or outside knowledge. It
may return fewer questions than requested when the selected material does not
support enough distinct questions.

Use **Upload lessons** on the Generate Practice Questions page to add multiple
lessons. Each upload is extracted locally; adding or removing a lesson
automatically regenerates one batch from all listed lessons only. Lesson files
are saved in the local uploads folder and deduplicated by file content.
Questions cite the lesson filename and the exact supporting passage. Exams and
training rows are not question-content sources in lesson mode. Exam passages
that the selected classifier labels with sufficient confidence can augment
topic classification; low-confidence lesson passages are labeled
**Unclassified**.

The **Use exam files for style** option is on by default. It extracts common
question-stem patterns, topic associations, and their frequency from local
exam PDFs, then reuses generic wording patterns to vary question stems. This is
pattern reuse, not generative learning: exam facts, answers, and distractors
are not copied into lesson questions. The page shows how many exam files are
used as style references separately from the number of lesson content sources.
Classifier fitting happens on demand during analysis or generation and is
cached in memory; generation reports the new fit time and number of confidently
labeled exam passages added.

Topic likelihood uses TF-IDF emphasis, headings, definitions, lists,
emphasized terms, repetition, and (when available) past-exam frequency,
question wording similarity, and recent saved-practice coverage. Without
readable past exams, ranking uses lesson emphasis only. Each topic and question
shows a qualitative likelihood label and its evidence, not a probability or
guarantee. Scanned PDFs need the separately installed local Tesseract OCR
engine.

Every generated item shows its topic, requested difficulty, topic-priority
score, model relevance score, source dataset, source material, and the source
text used. Answers are withheld until **Show Answers** is selected. Saved
questions are persisted in the local `generated_questions` SQLite table and
are omitted from the normal saved-question response unless an answer-key
request is explicitly made. Open **Saved Questions** in the menu to review
saved sets grouped by generation session; sessions can be renamed, deleted,
expanded, or collapsed, with answers revealed per session or all at once.
CSV export leaves answer cells
blank until the answer key is revealed; PDF export opens the system print
dialog.

The configurable priority score is:

```text
selected-dataset topic frequency × configured weight
+ course-material importance × configured weight
+ model topic confidence × configured weight
+ recent occurrence × configured weight
```

Weights must total 100%. The current training-example schema has no dates, so
recent occurrence is explicitly assigned a neutral 50% component rather than
inventing a chronology. The score measures priority/relevance in selected
data; it does not predict or guarantee which topics or questions will appear
on an actual examination. Dataset labels and records are user-provided and are
not automatically verified as official historical examination data.

Question candidates are checked for valid topic/type/difficulty, cited-passage
answer support, distractor integrity, and character n-gram similarity against
generated/saved questions and question-like training examples. The similarity
threshold is configurable. Source sentences below 45% selected-model relevance
are rejected to avoid attributing an unrelated source sentence to a training
topic. Multiple-choice distractors are restricted to the same topic, selected
in order from the same lesson, other uploaded lessons, then training
definitions when not in lesson mode; lesson-mode training data is never used
for distractors. Junk, stem-overlapping, cited-passage, and conflicting answer
choices are rejected. If too few distractors remain, the question is skipped
and counted in the diagnostics. The generator does not fill shortages with
random or unsupported content. Generated answers remain in the local
database; an answer key is returned only by the explicit reveal action.
The locally fitted TF-IDF/SVM/Naive Bayes ensemble is cached for reuse with
unchanged training examples where the bounded cache applies; larger datasets
are fit without retaining a large cache entry. During one question-generation
request, each distinct dataset variant is fit or retrieved once and scores
eligible passages together. Lesson generation can first use the selected
dataset to assign confidence-qualified topic labels to exam passages, then
fit or retrieve a classifier for the augmented examples.
Difficulty filtering uses a transparent sentence-length/complexity-cue
heuristic, not standardized exam-level calibration; the user-selected label is
shown on each question.

Rate generated or saved questions as **Good** or **Not for me**. Ratings are
stored locally and form one global preference profile across subjects. Future
question sets use the selected question type, difficulty, generic wording
patterns, question lengths, and answer lengths associated with those ratings
to adjust candidate ordering; feedback does not train the topic classifier,
change source validation, or invent content. Ratings for a question can be
changed, and **Reset preferences** clears all saved ratings.

## Local storage and privacy

The desktop app stores data in:

```text
%LOCALAPPDATA%\Clinthinker©\storage\
```

On first launch after upgrading, the app moves your existing
`%LOCALAPPDATA%\Topic Prediction Assistant\storage\` data folder to the new
Clinthinker© location so your saved files and datasets remain available. App
data stays on this device. The PWA precaches its interface for offline use; to
upload files and make predictions, the local Python service must be running
(the desktop app starts and stops it automatically).
Clinthinker checks the configured GitHub Pages `version.json` once at startup
and only when **Check for updates** is selected in Settings. The check sends
only that manifest request; an installer is downloaded only after selecting
**Update**. The backend accepts installer URLs only from this project's GitHub
repository and uses the URL from its own fetched manifest. A manifest may
include a `sha256` field; when present, the installer must match it before
launch. Before launching the installer, the app creates a timestamped backup
of `assistant.sqlite3` in the storage folder.

When running from source, local app data is stored in
`backend\storage\`; desktop builds use the LocalAppData folder above.

The executable icon and web/PWA icons are generated from the supplied image
and stored in `assets/` and `frontend/public/`.

## Tests

Run the backend regression and practice-question tests from `backend`:

```powershell
python -m unittest discover -s tests -v
```

Build the desktop executable and frontend from `frontend`:

```powershell
npm run build
```

For a frontend-only build, use `npm run build:web`.
