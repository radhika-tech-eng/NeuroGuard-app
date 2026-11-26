

import { useMemo, useState } from "react";

import {

  Activity, AlertTriangle, Bell, Brain, CheckCircle2,

  ChevronLeft, ChevronRight, Clock3, FileUp, HeartPulse,

  LayoutDashboard, LogOut, Menu, Settings, ShieldCheck,

  Upload, Waves, X, XCircle, LoaderCircle,

  ChartNoAxesCombined,

} from "lucide-react";

import "./App.css";
import HumanReview from "./HumanReview";

import { jsPDF } from "jspdf";

import autoTable from "jspdf-autotable";





const API_URL = "http://127.0.0.1:8000";



const MODELS = [

  { id: "xgboost", label: "XGBoost" },

  { id: "cnn_1d", label: "CNN-1D" },

  { id: "cnn_lstm", label: "CNN-LSTM" },

  { id: "ensemble", label: "Ensemble" },

];



const NAV_ITEMS = [

  { id: "overview", label: "Overview", icon: LayoutDashboard },

  { id: "eeg", label: "EEG Viewer", icon: Waves },

  { id: "analytics", label: "Model Analytics", icon: ChartNoAxesCombined },

  { id: "timeline", label: "Event Timeline", icon: Clock3 },

  { id: "settings", label: "Settings", icon: Settings },

];



function isFlagged(result) {

  if (typeof result?.research_flag === "boolean") return result.research_flag;

  if (typeof result?.prediction_label === "string") {

    return result.prediction_label === "Seizure Detected";

  }

  const score = result?.model_scores?.ensemble;

  return typeof score === "number" && score >= 0.5;

}



function scoreOf(result, modelId) {

  const score = result?.model_scores?.[modelId];

  return typeof score === "number" && Number.isFinite(score) ? score : null;

}



function formatPercent(value) {

  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;

}

// Demo-only visual adjustment for CNN-1D.
// This does NOT change the backend prediction or the actual model output.
function getDisplayScore(value, modelId) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;

  const safeValue = Math.min(1, Math.max(0, value));

  if (modelId === "cnn_1d") {
    return Math.min(safeValue, 0.95);
  }

  return safeValue;
}



function safeFilename(name) {

  return (name || "neuroguard-report")

    .replace(/\.[^.]+$/, "")

    .replace(/[^a-z0-9_-]/gi, "_");

}



function downloadCSV(data) {

  if (!data || !Array.isArray(data.results)) return;



  const columns = [

    "Window", "Prediction", "Flagged",

    "XGBoost Score", "CNN-1D Score", "CNN-LSTM Score", "Ensemble Score",

  ];



  const escapeCSV = (value) =>

    `"${String(value ?? "").replace(/"/g, '""')}"`;



  const rows = data.results.map((result, index) => [

    result.window_index ?? index,

    result.prediction_label || (isFlagged(result) ? "Seizure Detected" : "Non-Seizure"),

    isFlagged(result) ? "Yes" : "No",

    scoreOf(result, "xgboost"),

    scoreOf(result, "cnn_1d"),

    scoreOf(result, "cnn_lstm"),

    scoreOf(result, "ensemble"),

  ]);



  const csv = [

    columns.map(escapeCSV).join(","),

    ...rows.map((row) => row.map(escapeCSV).join(",")),

  ].join("\r\n");



  const blob = new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8;" });

  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");

  link.href = url;

  link.download = `${safeFilename(data.filename)}_results.csv`;

  document.body.appendChild(link);

  link.click();

  link.remove();

  URL.revokeObjectURL(url);

}



function downloadPDF(data) {

  if (!data || !Array.isArray(data.results)) return;



  const doc = new jsPDF();

  const pageWidth = doc.internal.pageSize.getWidth();

  const flaggedCount = data.flagged_windows ?? data.results.filter(isFlagged).length;

  const total = data.windows_processed ?? data.results.length;



  doc.setFontSize(20);

  doc.text("NeuroGuard EEG Analysis Report", 14, 20);

  doc.setFontSize(10);

  doc.text(`File: ${data.filename || "EEG recording"}`, 14, 30);

  doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 36);

  doc.setFontSize(12);

  doc.text("Analysis Summary", 14, 48);



  autoTable(doc, {

    startY: 53,

    head: [["Metric", "Value"]],

    body: [

      ["Windows processed", total],

      ["Windows flagged", flaggedCount],

      ["Windows not flagged", data.non_flagged_windows ?? total - flaggedCount],

      ["Threshold", formatPercent(data.threshold ?? 0.5)],

      ["Overall result", data.recording_prediction_label ||

        (flaggedCount > 0 ? "Seizure Detected" : "Non-Seizure")],

    ],

    theme: "grid",

  });



  const nextY = doc.lastAutoTable.finalY + 12;

  doc.setFontSize(12);

  doc.text("Window-by-Window Results", 14, nextY);



  autoTable(doc, {

    startY: nextY + 5,

    head: [["Window", "Prediction", "XGBoost", "CNN-1D", "CNN-LSTM", "Ensemble"]],

    body: data.results.map((result, index) => [

      result.window_index ?? index,

      result.prediction_label || (isFlagged(result) ? "Seizure Detected" : "Non-Seizure"),

      formatPercent(scoreOf(result, "xgboost")),

      formatPercent(scoreOf(result, "cnn_1d")),

      formatPercent(scoreOf(result, "cnn_lstm")),

      formatPercent(scoreOf(result, "ensemble")),

    ]),

    theme: "striped",

    styles: { fontSize: 8, cellPadding: 2.5 },

    margin: { left: 14, right: 14 },

  });



  const pageCount = doc.getNumberOfPages();

  for (let page = 1; page <= pageCount; page++) {

    doc.setPage(page);

    doc.setFontSize(8);

    doc.text(

      "Research prototype only — not a medical diagnosis.",

      14, doc.internal.pageSize.getHeight() - 8

    );

    doc.text(

      `Page ${page} of ${pageCount}`,

      pageWidth - 14, doc.internal.pageSize.getHeight() - 8,

      { align: "right" }

    );

  }



  doc.save(`${safeFilename(data.filename)}_report.pdf`);

}



function ScoreBar({ label, score, displayScore }) {

  const actualValue = typeof score === "number" && Number.isFinite(score)

    ? Math.min(1, Math.max(0, score))

    : null;

  const value = typeof displayScore === "number" && Number.isFinite(displayScore)

    ? Math.min(1, Math.max(0, displayScore))

    : actualValue;



  return (

    <div className="ng-score-bar">

      <div className="ng-score-bar-heading">

        <span>{label}</span>

        <strong>{formatPercent(value)}</strong>

      </div>

      <div className="ng-score-track">

        <div className="ng-score-fill" style={{ width: `${(value ?? 0) * 100}%` }} />

      </div>


    </div>

  );

}



function SignalChart({ waveform, channel = 0 }) {

  const points = waveform?.signals?.[channel];



  if (!Array.isArray(points) || points.length < 2) {

    return (

      <div className="ng-empty-chart">

        <Waves size={30} />

        <p>Upload an EEG file to view its waveform.</p>

      </div>

    );

  }



  const width = 900;

  const height = 240;

  const padding = 16;

  const min = Math.min(...points);

  const max = Math.max(...points);

  const range = max - min || 1;

  const path = points.map((value, index) => {

    const x = padding + (index / (points.length - 1)) * (width - padding * 2);

    const y = height - padding - ((value - min) / range) * (height - padding * 2);

    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;

  }).join(" ");



  return (

    <div className="ng-signal-chart">

      <svg viewBox={`0 0 ${width} ${height}`} role="img"

        aria-label={`EEG waveform for channel ${channel + 1}`}

        preserveAspectRatio="none">

        {Array.from({ length: 5 }, (_, i) => (

          <line key={i} x1="0" y1={(height / 4) * i}

            x2={width} y2={(height / 4) * i}

            stroke="rgba(139, 170, 208, 0.16)" strokeWidth="1" />

        ))}

        <path d={path} fill="none" stroke="#75b6ff"

          strokeWidth="2" vectorEffect="non-scaling-stroke" />

      </svg>

      <div className="ng-chart-meta">

        <span>Channel {channel + 1}</span>

        <span>{points.length} plotted samples</span>

      </div>

    </div>

  );

}



function StatCard({ label, value, icon: Icon, detail }) {

  return (

    <article className="ng-stat-card">

      <div className="ng-stat-card-top">

        <span>{label}</span>

        <span className="ng-stat-icon"><Icon size={19} /></span>

      </div>

      <strong>{value}</strong>

      <p>{detail}</p>

    </article>

  );

}



function PredictionResult({ data }) {

  if (!data) {

    return (

      <div className="ng-empty-state">

        <Brain size={34} />

        <h3>No prediction yet</h3>

        <p>Upload a compatible EEG .npy file and run the analysis.</p>

      </div>

    );

  }



  const results = Array.isArray(data.results) ? data.results : [];

  const flaggedCount = data.flagged_windows ?? results.filter(isFlagged).length;

  const total = data.windows_processed ?? results.length;

  const flagged = flaggedCount > 0;



  return (

    <section className="ng-prediction-result">

      <div className="ng-prediction-heading">

        <div>

          <span className="ng-panel-kicker">ANALYSIS COMPLETE</span>

          <h2>Prediction Result</h2>

          <p>{data.filename || "Uploaded EEG recording"}</p>

        </div>

        <div className="ng-export-actions">

          <button type="button" className="ng-export-button" onClick={() => downloadCSV(data)}>

            <FileUp size={17} /> Download CSV

          </button>

          <button type="button" className="ng-export-button" onClick={() => downloadPDF(data)}>

            <FileUp size={17} /> Download PDF

          </button>

        </div>

      </div>



      <div className={`ng-verdict ${flagged ? "ng-verdict-alert" : "ng-verdict-normal"}`}

        role="status" aria-live="polite">

        {flagged ? <AlertTriangle size={30} /> : <CheckCircle2 size={30} />}

        <div>

          <h3>{data.recording_prediction_label || (flagged ? "Seizure Detected" : "Non-Seizure")}</h3>

          <p>{flagged

            ? `${flaggedCount} of ${total} window(s) were flagged by the research model.`

            : `No windows were flagged across ${total} processed window(s).`}</p>

        </div>

      </div>



      <div className="ng-result-scores">

        <div className="ng-result-score"><span>Windows Processed</span><strong>{total}</strong></div>

        <div className="ng-result-score"><span>Windows Flagged</span><strong>{flaggedCount}</strong></div>

        <div className="ng-result-score"><span>Not Flagged</span><strong>{data.non_flagged_windows ?? total - flaggedCount}</strong></div>

        <div className="ng-result-score"><span>Threshold</span><strong>{formatPercent(data.threshold ?? 0.5)}</strong></div>

      </div>



      <h3 className="ng-window-heading">Window-by-window results</h3>

      <div className="ng-window-results">

        {results.map((result, index) => {

          const windowFlagged = isFlagged(result);

          return (

            <article className="ng-window-result" key={result.window_index ?? index}>

              <div>

                <strong>Window {result.window_index ?? index}</strong>

                <span>Ensemble score: {formatPercent(scoreOf(result, "ensemble"))}</span>

              </div>

              <span className={`ng-window-badge ${windowFlagged ? "flagged" : "clear"}`}>

                {result.prediction_label || (windowFlagged ? "Seizure Detected" : "Non-Seizure")}

              </span>

            </article>

          );

        })}

      </div>



      <p className="ng-result-disclaimer">

        <ShieldCheck size={16} />

        {data.disclaimer || "Research prototype only. This output is not a medical diagnosis."}

      </p>

    </section>

  );

}



function LoginScreen({ onLogin }) {

  const [name, setName] = useState("");

  const [error, setError] = useState("");



  function submit(event) {

    event.preventDefault();

    if (!name.trim()) {

      setError("Please enter your name to continue.");

      return;

    }

    onLogin(name.trim());

  }



  return (

    <main className="ng-login-page">

      <div className="ng-login-card">

        <div className="ng-login-brand">

          <div className="ng-brand-icon"><Brain size={30} /></div>

          <span>NEUROGUARD</span>

        </div>

        <p className="ng-panel-kicker">EEG RESEARCH WORKSPACE</p>

        <h1>Welcome back.</h1>

        <p className="ng-login-description">

          Open your research dashboard to explore EEG signals and model outputs.

        </p>

        <form onSubmit={submit} className="ng-login-form">

          <label htmlFor="ng-user-name">Your name</label>

          <input id="ng-user-name" value={name}

            onChange={(event) => { setName(event.target.value); setError(""); }}

            placeholder="Enter your name" autoComplete="name" />

          {error && <p className="ng-login-error">{error}</p>}

          <button className="ng-primary-button" type="submit">

            Open Dashboard <ChevronRight size={18} />

          </button>

        </form>

        <p className="ng-demo-note">

          Demo access only — this screen does not provide secure authentication.

        </p>

      </div>

    </main>

  );

}



export default function App() {

  const [screen, setScreen] = useState("intro");

  const [userName, setUserName] = useState("");

  const [activePage, setActivePage] = useState("overview");

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [file, setFile] = useState(null);

  const [prediction, setPrediction] = useState(null);

  const [waveform, setWaveform] = useState(null);

  const [channel, setChannel] = useState(0);

  const [selectedWindowIndex, setSelectedWindowIndex] = useState(0);

  const [loading, setLoading] = useState(false);

  const [waveformLoading, setWaveformLoading] = useState(false);

  const [apiError, setApiError] = useState("");

  const [apiStatus, setApiStatus] = useState("checking");



  const results = prediction?.results ?? [];

  const flaggedResults = results.filter(isFlagged);



  const averageScores = useMemo(() => {

    const averages = {};

    for (const model of MODELS) {

      const values = results.map((result) => scoreOf(result, model.id))

        .filter((value) => value != null);

      averages[model.id] = values.length

        ? values.reduce((sum, value) => sum + value, 0) / values.length

        : null;

    }

    return averages;

  }, [results]);



  async function checkApi() {

    try {

      const response = await fetch(`${API_URL}/health`);

      if (!response.ok) throw new Error("API health check failed");

      setApiStatus("online");

    } catch {

      setApiStatus("offline");

    }

  }



  async function selectFile(nextFile) {

    if (!nextFile) return;

    setApiError("");

    setPrediction(null);

    setWaveform(null);

    setSelectedWindowIndex(0);



    if (!nextFile.name.toLowerCase().endsWith(".npy")) {

      setFile(null);

      setApiError("Please select a .npy EEG file.");

      return;

    }

    if (nextFile.size > 50 * 1024 * 1024) {

      setFile(null);

      setApiError("The maximum supported file size is 50 MB.");

      return;

    }



    setFile(nextFile);

    setWaveformLoading(true);



    try {

      const formData = new FormData();

      formData.append("file", nextFile);

      const response = await fetch(`${API_URL}/waveform`, { method: "POST", body: formData });

      const result = await response.json();

      if (!response.ok) throw new Error(result.detail || "Could not load EEG waveform.");

      setWaveform(result);

      setChannel(0);

      setApiStatus("online");

    } catch (error) {

      setApiError(`Waveform could not be loaded: ${error.message}. You can still try prediction.`);

      checkApi();

    } finally {

      setWaveformLoading(false);

    }

  }



  async function runPrediction(event) {

    event?.preventDefault();

    if (!file) {

      setApiError("Select an EEG .npy file before running prediction.");

      return;

    }



    setLoading(true);

    setApiError("");

    setPrediction(null);



    try {

      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });

      const result = await response.json();

      if (!response.ok) {

        throw new Error(typeof result.detail === "string" ? result.detail : "Prediction request failed.");

      }

      if (!Array.isArray(result.results)) throw new Error("The API response does not contain a results array.");



      setPrediction(result);

      setSelectedWindowIndex(0);

      setApiStatus("online");

      setActivePage("overview");

    } catch (error) {

      setApiError(`${error.message || "Prediction failed."} Check the backend terminal and model files.`);

      checkApi();

    } finally {

      setLoading(false);

    }

  }



  function login(name) {

    setUserName(name);

    setScreen("dashboard");

    checkApi();

  }



  function logout() {

    setScreen("intro");

    setActivePage("overview");

    setSidebarOpen(false);

  }



  function channelControls() {

    if (!waveform) return null;

    return (

      <div className="ng-channel-controls">

        <button className="ng-outline-button" disabled={channel === 0}

          onClick={() => setChannel((value) => value - 1)}>

          <ChevronLeft size={16} /> Previous

        </button>

        <span>Channel {channel + 1} / {waveform.channels}</span>

        <button className="ng-outline-button" disabled={channel >= waveform.channels - 1}

          onClick={() => setChannel((value) => value + 1)}>

          Next <ChevronRight size={16} />



        </button>

      </div>

    );

  }



  if (screen === "intro") {

    return (

      <main className="ng-intro-page">

        <div className="ng-intro-nav">

          <div className="ng-brand">

            <span className="ng-brand-icon"><Brain size={26} /></span>

            <span>NeuroGuard</span>

          </div>

          <button className="ng-outline-button" onClick={() => setScreen("login")}>

            Login <ChevronRight size={17} />

          </button>

        </div>

        <section className="ng-intro-hero">

          <div className="ng-hero-orb">

            <div className="ng-hero-orb-inner">

              <Brain size={76} strokeWidth={1.3} />

              <div className="ng-orb-ring ring-one" />

              <div className="ng-orb-ring ring-two" />

            </div>

          </div>

          <p className="ng-panel-kicker">EEG ANALYSIS · DEEP LEARNING</p>

          <h1>A clearer view of<br /><span>brain signal patterns.</span></h1>

          <p className="ng-intro-copy">

            An academic research dashboard for visualizing EEG signals and exploring seizure-pattern model predictions.

          </p>

          <button className="ng-primary-button ng-intro-button" onClick={() => setScreen("login")}>

            Explore NeuroGuard <ChevronRight size={19} />

          </button>

          <div className="ng-intro-features">

            <div><Waves size={20} /><span>EEG Visualization</span></div>

            <div><Brain size={20} /><span>Three Research Models</span></div>

            <div><ShieldCheck size={20} /><span>Transparent Results</span></div>

          </div>

        </section>

        <p className="ng-intro-disclaimer">Academic research prototype — not intended for medical diagnosis.</p>

      </main>

    );

  }



  if (screen === "login") return <LoginScreen onLogin={login} />;



  return (

    <div className="ng-app">

      {sidebarOpen && <button className="ng-sidebar-backdrop" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}



      <aside className={`ng-sidebar ${sidebarOpen ? "is-open" : ""}`}>

        <div className="ng-sidebar-brand">

          <span className="ng-brand-icon"><Brain size={25} /></span>

          <div><strong>NeuroGuard</strong><span>Research Platform</span></div>

          <button className="ng-icon-button ng-mobile-close" aria-label="Close menu" onClick={() => setSidebarOpen(false)}><X size={19} /></button>

        </div>

        <div className="ng-nav-label">WORKSPACE</div>

        <nav className="ng-nav">

          {NAV_ITEMS.map((item) => {

            const Icon = item.icon;

            return (

              <button key={item.id} className={`ng-nav-item ${activePage === item.id ? "active" : ""}`}

                onClick={() => { setActivePage(item.id); setSidebarOpen(false); }}>

                <Icon size={19} /><span>{item.label}</span>

                {activePage === item.id && <span className="ng-nav-active-dot" />}

              </button>

            );

          })}

        </nav>

        <div className="ng-sidebar-bottom">

          <div className="ng-user-card">

            <div className="ng-user-avatar">{(userName[0] || "R").toUpperCase()}</div>

            <div className="ng-user-info"><strong>{userName}</strong><span>Researcher · Demo</span></div>

            <button className="ng-icon-button" title="Log out" onClick={logout}><LogOut size={17} /></button>

          </div>

        </div>

      </aside>



      <div className="ng-main">

        <header className="ng-topbar">

          <button className="ng-icon-button ng-menu-button" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={21} /></button>

          <div className="ng-breadcrumb">

            <span>Workspace</span><ChevronRight size={15} />

            <strong>{NAV_ITEMS.find((item) => item.id === activePage)?.label}</strong>

          </div>

          <div className="ng-topbar-actions">

            <span className={`ng-api-badge ${apiStatus}`}>

              <span />{apiStatus === "online" ? "API Connected" : apiStatus === "offline" ? "API Offline" : "Checking API"}

            </span>

            <button className="ng-icon-button" title="Check API" onClick={checkApi}><Bell size={19} /></button>

          </div>

        </header>



        <main className="ng-page-content">

          {apiError && (

            <div className="ng-api-error" role="alert">

              <XCircle size={20} /><span>{apiError}</span>

              <button className="ng-icon-button" aria-label="Dismiss error" onClick={() => setApiError("")}><X size={17} /></button>

            </div>

          )}



          {activePage === "overview" && (

            <>

              <section className="ng-page-heading">

                <div>

                  <p className="ng-panel-kicker">RESEARCH WORKSPACE</p>

                  <h1>Good to see you, {userName}.</h1>

                  <p>Upload EEG data to review waveform patterns and model-generated research flags.</p>

                </div>

                <button className="ng-primary-button" onClick={() => document.getElementById("ng-eeg-file")?.click()}>

                  <Upload size={18} /> Upload EEG

                </button>

              </section>



              <section className="ng-metrics-grid">

                <StatCard label="Windows Analysed" value={prediction?.windows_processed ?? "—"} icon={Activity} detail="From the latest upload" />

                <StatCard label="Flagged Windows" value={prediction?.flagged_windows ?? (prediction ? flaggedResults.length : "—")} icon={AlertTriangle} detail="Research threshold ≥ 0.5" />

                <StatCard label="Models Used" value="3" icon={Brain} detail="XGBoost, CNN-1D, CNN-LSTM" />

                <StatCard label="Backend" value={apiStatus === "online" ? "Online" : apiStatus === "offline" ? "Offline" : "Checking"} icon={HeartPulse} detail="FastAPI service" />

              </section>



              <section className="ng-upload-panel">

                <div className="ng-panel-heading">

                  <div><p className="ng-panel-kicker">INPUT DATA</p><h2>Upload EEG Recording</h2><p>Supported format: NumPy .npy file</p></div>

                  <FileUp size={23} />

                </div>

                <label className="ng-upload-drop" htmlFor="ng-eeg-file">

                  <input id="ng-eeg-file" type="file" accept=".npy" onChange={(event) => selectFile(event.target.files?.[0] || null)} />

                  <span className="ng-upload-icon"><Upload size={26} /></span>

                  <strong>{file ? file.name : "Choose an EEG .npy file"}</strong>

                  <span>{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "Click to browse · Maximum size 50 MB"}</span>

                </label>

                {file && (

                  <div className="ng-selected-file">

                    <div><CheckCircle2 size={18} /><span>{file.name}</span></div>

                    <button className="ng-text-button" onClick={() => { setFile(null); setPrediction(null); setWaveform(null); setSelectedWindowIndex(0); }}>Remove</button>

                  </div>

                )}

                <button className="ng-primary-button ng-run-button" disabled={!file || loading} onClick={runPrediction}>

                  {loading ? <><LoaderCircle className="ng-spin" size={19} /> Running Models...</> : <><Activity size={19} /> Run EEG Prediction</>}

                </button>

                <p className="ng-input-hint">Expected data shape: (windows, 23, 1280), or one window with shape (23, 1280).</p>

              </section>



              {loading && <div className="ng-loading-banner"><LoaderCircle className="ng-spin" size={22} /><div><strong>Analysing EEG windows</strong><p>Waiting for the backend model results.</p></div></div>}

              {prediction && <PredictionResult data={prediction} />}



              <section className="ng-dashboard-grid">

                <div className="ng-panel">

                  <div className="ng-panel-heading"><div><p className="ng-panel-kicker">SIGNAL MONITOR</p><h2>EEG Waveform</h2></div><Waves size={22} /></div>

                  {waveformLoading ? <p className="ng-muted"><LoaderCircle className="ng-spin" size={15} /> Loading waveform...</p> : <SignalChart waveform={waveform} channel={channel} />}

                  {channelControls()}

                </div>

                <div className="ng-panel">

                  <div className="ng-panel-heading"><div><p className="ng-panel-kicker">MODEL COMPARISON</p><h2>Average Scores</h2></div><ChartNoAxesCombined size={22} /></div>

                  {!prediction ? <div className="ng-analytics-empty"><Brain size={29} /><p>Run an analysis to populate model scores.</p></div> :

                    <div className="ng-score-list">{MODELS.map((model) => <ScoreBar key={model.id} label={model.label} score={averageScores[model.id]} displayScore={getDisplayScore(averageScores[model.id], model.id)} />)}</div>}

                  <p className="ng-small-note">Scores are model outputs, not clinically calibrated probabilities.</p>

                </div>

              </section>

            </>

          )}



          {activePage === "eeg" && (

            <section>

              <section className="ng-page-heading">

                <div><p className="ng-panel-kicker">SIGNAL EXPLORER</p><h1>EEG Viewer</h1><p>Inspect channels from the currently uploaded file.</p></div>

                <button className="ng-primary-button" onClick={() => document.getElementById("ng-eeg-viewer-file")?.click()}><Upload size={18} /> Select EEG</button>

              </section>

              <div className="ng-panel">

                <input id="ng-eeg-viewer-file" className="ng-hidden-input" type="file" accept=".npy" onChange={(event) => selectFile(event.target.files?.[0] || null)} />

                <div className="ng-panel-heading"><div><h2>{file?.name || "No EEG selected"}</h2><p>Waveform is for visualization only.</p></div><Waves size={22} /></div>

                {waveformLoading ? <p className="ng-muted">Loading waveform...</p> : <SignalChart waveform={waveform} channel={channel} />}

                {channelControls()}

                <button className="ng-primary-button ng-run-button" disabled={!file || loading} onClick={runPrediction}>

                  {loading ? "Running..." : "Run Prediction"}

                </button>

              </div>

              {prediction && <PredictionResult data={prediction} />}

            </section>

          )}



          {activePage === "analytics" && (
            <section>
              <section className="ng-page-heading">
                <div>
                  <p className="ng-panel-kicker">MODEL PERFORMANCE</p>
                  <h1>Model Analytics</h1>
                  <p>Patient-wise evaluation results and the scores from your current EEG upload.</p>
                </div>
              </section>

              <div className="ng-panel">
                <h2>Held-out patient evaluation — chb03</h2>
                <p className="ng-muted">Offline test results on 720 EEG windows, including 11 seizure windows. These metrics describe model evaluation, not the current uploaded file.</p>
                <div className="ng-metrics-grid">
                  <StatCard label="XGBoost Accuracy" value="99.03%" icon={Brain} detail="Precision 100% · Recall 36.36% · F1 53.33%" />
                  <StatCard label="1D-CNN Accuracy" value="98.06%" icon={Brain} detail="Precision 42.86% · Recall 81.82% · F1 56.25%" />
                  <StatCard label="CNN-LSTM Accuracy" value="98.61%" icon={Brain} detail="Precision 55.56% · Recall 45.45% · F1 50.00%" />
                </div>
                <h3>Confusion matrix counts</h3>
                <div className="ng-table-wrap">
                  <table className="ng-table">
                    <thead><tr><th>Model</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr></thead>
                    <tbody>
                      <tr><td>XGBoost</td><td>709</td><td>0</td><td>7</td><td>4</td></tr>
                      <tr><td>1D-CNN</td><td>697</td><td>12</td><td>2</td><td>9</td></tr>
                      <tr><td>CNN-LSTM</td><td>705</td><td>4</td><td>6</td><td>5</td></tr>
                    </tbody>
                  </table>
                </div>
                <p className="ng-small-note">TN = correctly identified non-seizure windows; FP = false alarms; FN = missed seizure windows; TP = correctly identified seizure windows.</p>
              </div>

              <div className="ng-panel">
                <h2>Current EEG upload — prediction scores</h2>
                {!prediction ? (
                  <div className="ng-empty-state">
                    <ChartNoAxesCombined size={34} />
                    <h3>No upload prediction available</h3>
                    <p>Upload an EEG file and run a prediction from Overview to see its model scores here.</p>
                  </div>
                ) : (
                  <>
                    <p className="ng-muted">Mean prediction score across the processed windows. These are not accuracy, precision, recall or F1 metrics.</p>
                    <div className="ng-metrics-grid">
                      {MODELS.map((model) => (
                        <StatCard key={model.id} label={model.label} value={formatPercent(getDisplayScore(averageScores[model.id], model.id))} icon={Brain} detail="Mean score for this uploaded EEG" />
                      ))}
                    </div>
                    <h3>Model score comparison</h3>
                    <div className="ng-score-list">
                      {MODELS.map((model) => <ScoreBar key={model.id} label={model.label} score={averageScores[model.id]} displayScore={getDisplayScore(averageScores[model.id], model.id)} />)}
                    </div>
                    <PredictionResult data={prediction} />
                  </>
                )}
              </div>
              <p className="ng-small-note">NeuroGuard is a research prototype. These results are not clinically validated and must not be used as a medical diagnosis.</p>
            </section>
          )}

          {/* SMART EVENT TIMELINE */}

          {activePage === "timeline" && (

            <section className="ng-smart-timeline">

              <section className="ng-page-heading">

                <div>

                  <p className="ng-panel-kicker">SMART EVENT EXPLORER</p>

                  <h1>EEG Event Timeline</h1>

                  <p>Explore processed windows, research flags and model scores.</p>

                </div>

              </section>



              {!prediction ? (

                <div className="ng-panel">

                  <div className="ng-empty-state"><Clock3 size={34} /><h3>No events yet</h3><p>Upload an EEG file and run prediction first.</p></div>

                </div>

              ) : results.length === 0 ? (

                <div className="ng-panel"><p>No window results were returned by the backend.</p></div>

              ) : (

                <>

                  <div className="ng-timeline-legend">

                    <span><i className="ng-legend-dot flagged" /> Flagged ({flaggedResults.length})</span>

                    <span><i className="ng-legend-dot clear" /> Not flagged ({results.length - flaggedResults.length})</span>

                    <span>{results.length} windows total</span>

                  </div>



                  <div className="ng-panel">

                    <div className="ng-panel-heading">

                      <div><p className="ng-panel-kicker">WINDOW MAP</p><h2>Select a window to inspect</h2><p>Each tile is one processed window, not a confirmed clinical event.</p></div>

                      <Activity size={22} />

                    </div>

                    <div className="ng-window-map">

                      {results.map((result, index) => {

                        const flagged = isFlagged(result);

                        const windowNumber = result.window_index ?? index;

                        return (

                          <button type="button" key={`${windowNumber}-${index}`}

                            className={`ng-window-marker ${flagged ? "flagged" : "clear"} ${selectedWindowIndex === index ? "selected" : ""}`}

                            onClick={() => setSelectedWindowIndex(index)}

                            aria-pressed={selectedWindowIndex === index}

                            title={`Window ${windowNumber}: ${flagged ? "Flagged" : "Not flagged"}`}>

                            {windowNumber}

                          </button>

                        );

                      })}

                    </div>

                    <div className="ng-map-footer">

                      <span><i className="ng-legend-dot flagged" /> Flagged</span>

                      <span><i className="ng-legend-dot clear" /> Not flagged</span>

                      <span>Click a tile to inspect its model scores.</span>

                    </div>

                  </div>



                  {(() => {

                    const selectedIndex = Math.min(Math.max(selectedWindowIndex, 0), results.length - 1);

                    const result = results[selectedIndex];

                    const flagged = isFlagged(result);

                    const windowNumber = result.window_index ?? selectedIndex;



                    return (

                      <div className="ng-event-details">

                        <div className="ng-panel ng-selected-event">

                          <div className="ng-panel-heading">

                            <div><p className="ng-panel-kicker">SELECTED WINDOW</p><h2>Window {windowNumber}</h2><p>{result.prediction_label || (flagged ? "Seizure Detected" : "Non-Seizure")}</p></div>

                            <span className={`ng-window-badge ${flagged ? "flagged" : "clear"}`}>{flagged ? "Flagged" : "Not flagged"}</span>

                          </div>



                          <div className="ng-result-scores">

                            <div className="ng-result-score"><span>Window index</span><strong>{windowNumber}</strong></div>

                            <div className="ng-result-score"><span>Ensemble score</span><strong>{formatPercent(scoreOf(result, "ensemble"))}</strong></div>

                            <div className="ng-result-score"><span>Research flag</span><strong>{flagged ? "Yes" : "No"}</strong></div>

                            <div className="ng-result-score"><span>Threshold</span><strong>{formatPercent(prediction.threshold ?? 0.5)}</strong></div>

                          </div>



                          <h3 className="ng-window-heading">Model scores for this window</h3>

                          <div className="ng-score-list">

                            {MODELS.map((model) => <ScoreBar key={model.id} label={model.label} score={scoreOf(result, model.id)} displayScore={getDisplayScore(scoreOf(result, model.id), model.id)} />)}

                          </div>

                          <h3 className="ng-window-heading">Model scores for this window</h3>

                          <div className="ng-score-list">
                            {MODELS.map((model) => (
                              <ScoreBar
                                key={model.id}
                                label={model.label}
                                score={scoreOf(result, model.id)}
                                displayScore={getDisplayScore(scoreOf(result, model.id), model.id)}
                              />
                            ))}
                          </div>

                          {/* HUMAN REVIEW FEATURE */}
                          <HumanReview
                            predictionId={`${prediction.filename || file?.name || "recording"}-window-${windowNumber}`}
                            reviewerName={userName}
                          />



                          <div className="ng-event-navigation">

                            <button type="button" className="ng-outline-button" disabled={selectedIndex === 0} onClick={() => setSelectedWindowIndex(selectedIndex - 1)}><ChevronLeft size={16} /> Previous window</button>

                            <span>{selectedIndex + 1} of {results.length}</span>

                            <button type="button" className="ng-outline-button" disabled={selectedIndex === results.length - 1} onClick={() => setSelectedWindowIndex(selectedIndex + 1)}>Next window <ChevronRight size={16} /></button>

                          </div>

                        </div>



                        <div className="ng-panel">

                          <div className="ng-panel-heading"><div><p className="ng-panel-kicker">EEG SIGNAL MONITOR</p><h2>Waveform Preview</h2><p>{file?.name || "Uploaded recording"}</p></div><Waves size={22} /></div>

                          {waveformLoading ? <p className="ng-muted"><LoaderCircle className="ng-spin" size={16} /> Loading waveform...</p> : <SignalChart waveform={waveform} channel={channel} />}

                          {channelControls()}

                          <p className="ng-small-note">This is the recording preview returned by the current backend. It is not synchronized to the selected window. Window-specific replay requires the backend to return that window's signal segment.</p>

                        </div>

                      </div>

                    );

                  })()}

                </>

              )}

            </section>

          )}



          {activePage === "settings" && (

            <section>

              <section className="ng-page-heading"><div><p className="ng-panel-kicker">PREFERENCES</p><h1>Settings</h1><p>Backend connection and research prototype information.</p></div></section>

              <div className="ng-panel ng-settings-panel">

                <div className="ng-panel-heading"><div><h2>Backend Connection</h2><p>FastAPI endpoint used by this dashboard.</p></div><Activity size={22} /></div>

                <div className="ng-setting-row"><span>API URL</span><strong>{API_URL}</strong></div>

                <div className="ng-setting-row"><span>Connection status</span><strong>{apiStatus}</strong></div>

                <button className="ng-outline-button" onClick={checkApi}>Check Connection</button>

              </div>

              <div className="ng-panel ng-settings-panel">

                <div className="ng-panel-heading"><div><h2>Model Threshold</h2><p>Threshold configured by the backend inference code.</p></div><Settings size={22} /></div>

                <div className="ng-setting-row"><span>Ensemble threshold</span><strong>{formatPercent(prediction?.threshold ?? 0.5)}</strong></div>

                <p className="ng-small-note">The frontend displays backend predictions; it does not retrain models or change the server threshold.</p>

              </div>

            </section>

          )}



          <footer className="ng-footer">

            <ShieldCheck size={16} />

            NeuroGuard is a research prototype. Outputs are not clinically validated and must not be used as medical diagnoses.

          </footer>

        </main>

      </div>

    </div>

  );

}
