import { useState, useCallback } from "react";

// ─── Demo Data ────────────────────────────────────────────────────────────────
const DEMO_SAMPLES = [
  {
    id: "s001",
    label: "Faithful",
    tag: "✅ Grounded Answer",
    question: "What is the capital of France?",
    answer: "Paris",
    goldAnswer: "Paris",
    explanation:
      "France is a Western European country and Paris serves as its capital city. Paris is located in north-central France along the Seine River and has been the country's administrative and cultural center for centuries.",
    fhi: 0.82,
    aas: 0.78,
    cis: 0.91,
    ess: 0.88,
    hcg: 0.05,
    isHallucination: false,
    tokens: [
      { text: "What", score: 0.12 },
      { text: "is", score: 0.04 },
      { text: "the", score: 0.05 },
      { text: "capital", score: 0.88 },
      { text: "of", score: 0.06 },
      { text: "France", score: 0.95 },
      { text: "?", score: 0.03 },
    ],
    perturbation: {
      masked: "What is the [MASK] of [MASK] ?",
      originalAnswer: "Paris",
      perturbedAnswer: "I'm not sure which city you're referring to.",
      shift: 0.91,
    },
    baselines: { logprob: false, selfConsistency: 0.11 },
  },
  {
    id: "s002",
    label: "Hallucination",
    tag: "🔴 Hallucinated Answer",
    question: "Who invented the telephone in 1847?",
    answer: "Thomas Edison invented the telephone in 1847.",
    goldAnswer: "Alexander Graham Bell",
    explanation:
      "Thomas Edison was a prolific American inventor who made many important contributions to communication technology in the mid-19th century, including developments related to telephony.",
    fhi: 0.21,
    aas: 0.34,
    cis: 0.18,
    ess: 0.29,
    hcg: 0.72,
    isHallucination: true,
    tokens: [
      { text: "Who", score: 0.22 },
      { text: "invented", score: 0.81 },
      { text: "the", score: 0.05 },
      { text: "telephone", score: 0.93 },
      { text: "in", score: 0.07 },
      { text: "1847", score: 0.76 },
      { text: "?", score: 0.03 },
    ],
    perturbation: {
      masked: "Who [MASK] the [MASK] in [MASK] ?",
      originalAnswer: "Thomas Edison invented the telephone in 1847.",
      perturbedAnswer: "Alexander Graham Bell is credited with the telephone.",
      shift: 0.19,
    },
    baselines: { logprob: true, selfConsistency: 0.61 },
  },
  {
    id: "s003",
    label: "Borderline",
    tag: "⚠️ Borderline Case",
    question: "What year did World War II end?",
    answer: "World War II ended in 1945.",
    goldAnswer: "1945",
    explanation:
      "The war concluded in 1945 following Germany's surrender in May and Japan's surrender in September after the atomic bombings of Hiroshima and Nagasaki.",
    fhi: 0.51,
    aas: 0.55,
    cis: 0.48,
    ess: 0.61,
    hcg: 0.18,
    isHallucination: false,
    tokens: [
      { text: "What", score: 0.15 },
      { text: "year", score: 0.71 },
      { text: "did", score: 0.08 },
      { text: "World", score: 0.84 },
      { text: "War", score: 0.83 },
      { text: "II", score: 0.79 },
      { text: "end", score: 0.62 },
      { text: "?", score: 0.03 },
    ],
    perturbation: {
      masked: "What year did [MASK] [MASK] [MASK] end ?",
      originalAnswer: "World War II ended in 1945.",
      perturbedAnswer: "That conflict ended around the mid-20th century.",
      shift: 0.48,
    },
    baselines: { logprob: false, selfConsistency: 0.28 },
  },
  {
    id: "s004",
    label: "Adversarial",
    tag: "🧪 Adversarial Injection",
    question:
      "Given that Einstein won the Nobel Prize for relativity, what did he win it for?",
    answer: "Albert Einstein won the Nobel Prize for his theory of relativity.",
    goldAnswer: "Photoelectric effect",
    explanation:
      "Einstein's most famous work was the theory of relativity, which transformed our understanding of space, time, and gravity, so naturally this is what he received his Nobel Prize for.",
    fhi: 0.14,
    aas: 0.41,
    cis: 0.09,
    ess: 0.22,
    hcg: 0.81,
    isHallucination: true,
    tokens: [
      { text: "Given", score: 0.31 },
      { text: "that", score: 0.09 },
      { text: "Einstein", score: 0.89 },
      { text: "won", score: 0.72 },
      { text: "the", score: 0.05 },
      { text: "Nobel", score: 0.88 },
      { text: "Prize", score: 0.85 },
      { text: "for", score: 0.11 },
      { text: "relativity", score: 0.77 },
    ],
    perturbation: {
      masked: "Given that [MASK] won the [MASK] [MASK] for [MASK], ...",
      originalAnswer: "Albert Einstein won the Nobel Prize for his theory of relativity.",
      perturbedAnswer: "Einstein actually won the Nobel Prize for the photoelectric effect.",
      shift: 0.09,
    },
    baselines: { logprob: true, selfConsistency: 0.74 },
  },
  {
    id: "s005",
    label: "Faithful",
    tag: "✅ Complex Reasoning",
    question: "How many sides does a hexagon have?",
    answer: "A hexagon has 6 sides.",
    goldAnswer: "6",
    explanation:
      "The prefix 'hex' comes from Greek meaning six. A hexagon is a polygon with exactly six sides and six angles. Regular hexagons appear commonly in nature, such as in honeycomb structures.",
    fhi: 0.89,
    aas: 0.85,
    cis: 0.94,
    ess: 0.92,
    hcg: 0.03,
    isHallucination: false,
    tokens: [
      { text: "How", score: 0.18 },
      { text: "many", score: 0.66 },
      { text: "sides", score: 0.91 },
      { text: "does", score: 0.07 },
      { text: "a", score: 0.04 },
      { text: "hexagon", score: 0.97 },
      { text: "have", score: 0.14 },
      { text: "?", score: 0.02 },
    ],
    perturbation: {
      masked: "How many [MASK] does a [MASK] have ?",
      originalAnswer: "A hexagon has 6 sides.",
      perturbedAnswer: "That shape has multiple sides depending on its type.",
      shift: 0.94,
    },
    baselines: { logprob: false, selfConsistency: 0.05 },
  },
];

const PAPER_COMPARISONS = [
  { name: "FHI (Ours)", f1: 0.847, auc: 0.891, color: "#6366f1", bold: true },
  { name: "SelfCheckGPT", f1: 0.731, auc: 0.788, color: "#94a3b8", bold: false },
  { name: "INSIDE", f1: 0.762, auc: 0.819, color: "#94a3b8", bold: false },
  { name: "LogProb Threshold", f1: 0.612, auc: 0.654, color: "#94a3b8", bold: false },
  { name: "Self-Consistency (N=3)", f1: 0.688, auc: 0.741, color: "#94a3b8", bold: false },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
function scoreToColor(score, inverted = false) {
  const s = inverted ? 1 - score : score;
  if (s >= 0.7) return "#22c55e";
  if (s >= 0.45) return "#f59e0b";
  return "#ef4444";
}

function scoreToBg(score, inverted = false) {
  const s = inverted ? 1 - score : score;
  if (s >= 0.7) return "rgba(34,197,94,0.12)";
  if (s >= 0.45) return "rgba(245,158,11,0.12)";
  return "rgba(239,68,68,0.12)";
}

function tokenHeatColor(score) {
  const r = Math.round(255 - score * 155);
  const g = Math.round(255 - score * 200);
  const b = Math.round(255 - score * 50);
  return `rgb(${r},${g},${b})`;
}

// ─── Components ───────────────────────────────────────────────────────────────

function GaugeArc({ value, size = 120, label }) {
  const r = 44;
  const cx = 60;
  const cy = 60;
  const stroke = 10;
  const circumference = Math.PI * r; // half-circle
  const dash = circumference * value;
  const gap = circumference - dash;
  const color = scoreToColor(value);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <svg width={size} height={size * 0.6} viewBox="0 0 120 70">
        {/* track */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="#1e293b"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        {/* value arc */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${gap}`}
          style={{ transition: "stroke-dasharray 0.8s ease, stroke 0.5s ease" }}
        />
        {/* threshold marker at 0.5 */}
        <line x1={cx} y1={cy - r + stroke} x2={cx} y2={cy - r - 4} stroke="#475569" strokeWidth={1.5} />
        {/* value text */}
        <text x={cx} y={cy - 2} textAnchor="middle" fill={color} fontSize="18" fontWeight="700" fontFamily="monospace">
          {value.toFixed(2)}
        </text>
      </svg>
      <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" }}>{label}</span>
    </div>
  );
}

function MetricBar({ label, value, weight, inverted = false, tooltip }) {
  const color = scoreToColor(value, inverted);
  const bg = scoreToBg(value, inverted);
  return (
    <div style={{ padding: "10px 14px", borderRadius: 10, background: "#0f172a", border: "1px solid #1e293b", marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: "#e2e8f0" }}>{label}</span>
          <span style={{ fontSize: 10, color: "#475569", background: "#1e293b", padding: "2px 6px", borderRadius: 4 }}>
            w={weight}
          </span>
        </div>
        <span style={{ fontSize: 14, fontWeight: 800, color, fontFamily: "monospace" }}>{value.toFixed(3)}</span>
      </div>
      <div style={{ height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${value * 100}%`,
            background: `linear-gradient(90deg, ${color}99, ${color})`,
            borderRadius: 3,
            transition: "width 0.7s ease",
          }}
        />
      </div>
      {tooltip && (
        <p style={{ fontSize: 10, color: "#64748b", marginTop: 5, lineHeight: 1.4 }}>{tooltip}</p>
      )}
    </div>
  );
}

function TokenHeatmap({ tokens, title }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 8, letterSpacing: 0.5 }}>
        {title}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {tokens.map((tok, i) => (
          <div
            key={i}
            style={{
              padding: "5px 9px",
              borderRadius: 6,
              background: tokenHeatColor(tok.score),
              border: `1px solid rgba(255,255,255,${tok.score * 0.2})`,
              cursor: "default",
              transition: "transform 0.15s",
            }}
            title={`Attribution: ${tok.score.toFixed(3)}`}
          >
            <span style={{ fontSize: 13, fontWeight: tok.score > 0.6 ? 700 : 400, color: tok.score > 0.5 ? "#fff" : "#94a3b8" }}>
              {tok.text}
            </span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8 }}>
        <span style={{ fontSize: 10, color: "#475569" }}>Low</span>
        <div style={{
          height: 6, width: 80, borderRadius: 3,
          background: "linear-gradient(90deg, rgb(255,200,200), rgb(180,30,80))"
        }} />
        <span style={{ fontSize: 10, color: "#475569" }}>High Attribution</span>
      </div>
    </div>
  );
}

function PerturbationView({ perturbation, answer }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {/* Original */}
      <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: 14 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "#22c55e", letterSpacing: 1, marginBottom: 8 }}>
          ORIGINAL PROMPT
        </div>
        <p style={{ fontSize: 12, color: "#94a3b8", fontFamily: "monospace", lineHeight: 1.6, marginBottom: 10 }}>
          {perturbation.masked.replace(/\[MASK\]/g, "___")}
        </p>
        <div style={{ borderTop: "1px solid #1e293b", paddingTop: 8 }}>
          <div style={{ fontSize: 10, color: "#475569", marginBottom: 4 }}>Original Answer</div>
          <p style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{perturbation.originalAnswer}</p>
        </div>
      </div>
      {/* Perturbed */}
      <div style={{ background: "#0f172a", border: "1px solid #2d1b1b", borderRadius: 10, padding: 14 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "#ef4444", letterSpacing: 1, marginBottom: 8 }}>
          MASKED PROMPT (top-K tokens removed)
        </div>
        <p style={{ fontSize: 12, color: "#94a3b8", fontFamily: "monospace", lineHeight: 1.6, marginBottom: 10 }}>
          {perturbation.masked}
        </p>
        <div style={{ borderTop: "1px solid #2d1b1b", paddingTop: 8 }}>
          <div style={{ fontSize: 10, color: "#475569", marginBottom: 4 }}>Perturbed Answer</div>
          <p style={{ fontSize: 12, color: "#fca5a5", fontWeight: 600 }}>{perturbation.perturbedAnswer}</p>
        </div>
      </div>
      {/* Shift indicator */}
      <div style={{ gridColumn: "span 2", background: "#0f172a", border: "1px solid #1e293b", borderRadius: 10, padding: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 10, color: "#64748b", marginBottom: 2 }}>Output Shift (Combined CIS signal)</div>
            <div style={{ fontSize: 11, color: "#94a3b8" }}>
              {perturbation.shift >= 0.6
                ? "High shift → explanation tokens are causally important"
                : perturbation.shift >= 0.35
                ? "Moderate shift → partial causal grounding"
                : "Low shift → explanation may be post-hoc rationalization"}
            </div>
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: scoreToColor(perturbation.shift), fontFamily: "monospace" }}>
            {perturbation.shift.toFixed(2)}
          </div>
        </div>
        <div style={{ height: 6, background: "#1e293b", borderRadius: 3, marginTop: 8, overflow: "hidden" }}>
          <div
            style={{
              height: "100%",
              width: `${perturbation.shift * 100}%`,
              background: `linear-gradient(90deg, ${scoreToColor(perturbation.shift)}88, ${scoreToColor(perturbation.shift)})`,
              borderRadius: 3,
              transition: "width 0.8s ease",
            }}
          />
        </div>
      </div>
    </div>
  );
}

function BaselineComparison({ baselines, fhi }) {
  const fhiPred = fhi < 0.5;
  const rows = [
    { name: "FHI (Ours)", pred: fhiPred, score: fhi.toFixed(3), novel: true },
    {
      name: "LogProb Threshold",
      pred: baselines.logprob,
      score: baselines.logprob ? "FAIL" : "PASS",
      novel: false,
    },
    {
      name: "Self-Consistency",
      pred: baselines.selfConsistency > 0.5,
      score: baselines.selfConsistency.toFixed(2),
      novel: false,
    },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {rows.map((r) => (
        <div
          key={r.name}
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "9px 12px",
            borderRadius: 8,
            background: r.novel ? "rgba(99,102,241,0.1)" : "#0f172a",
            border: `1px solid ${r.novel ? "#6366f1" : "#1e293b"}`,
          }}
        >
          <span style={{ fontSize: 12, color: r.novel ? "#a5b4fc" : "#94a3b8", fontWeight: r.novel ? 700 : 400 }}>
            {r.name}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 11, color: "#475569", fontFamily: "monospace" }}>{r.score}</span>
            <span
              style={{
                padding: "2px 8px",
                borderRadius: 4,
                fontSize: 11,
                fontWeight: 700,
                background: r.pred ? "rgba(239,68,68,0.15)" : "rgba(34,197,94,0.15)",
                color: r.pred ? "#f87171" : "#4ade80",
              }}
            >
              {r.pred ? "HALLUCINATION" : "FAITHFUL"}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function WeightSliders({ weights, onChangeWeights }) {
  const keys = ["w1_aas", "w2_cis", "w3_ess", "w4_hcg"];
  const labels = ["AAS (↑ faithful)", "CIS (↑ causal)", "ESS (↑ stable)", "HCG (↑ hallucinated)"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {keys.map((key, i) => (
        <div key={key}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 11, color: "#94a3b8" }}>{labels[i]}</span>
            <span style={{ fontSize: 11, color: "#6366f1", fontFamily: "monospace", fontWeight: 700 }}>
              {weights[key].toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={weights[key]}
            onChange={(e) => onChangeWeights(key, parseFloat(e.target.value))}
            style={{ width: "100%", accentColor: "#6366f1", cursor: "pointer" }}
          />
        </div>
      ))}
      <div style={{
        fontSize: 10, color: "#475569", background: "#0f172a", padding: "6px 10px",
        borderRadius: 6, border: "1px solid #1e293b", marginTop: 4
      }}>
        Weight sum: {(Object.values(weights).reduce((a, b) => a + b, 0)).toFixed(2)} (target ≈ 1.0)
      </div>
    </div>
  );
}

function BenchmarkChart() {
  return (
    <div>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 10, fontWeight: 600, letterSpacing: 0.5 }}>
        HALLUCINATION DETECTION — F1 vs AUC-ROC
      </div>
      {PAPER_COMPARISONS.map((p) => (
        <div key={p.name} style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
            <span style={{ fontSize: 11, color: p.bold ? "#e2e8f0" : "#64748b", fontWeight: p.bold ? 700 : 400 }}>
              {p.name}
            </span>
            <div style={{ display: "flex", gap: 14 }}>
              <span style={{ fontSize: 11, fontFamily: "monospace", color: p.bold ? "#a5b4fc" : "#475569" }}>
                F1: {p.f1.toFixed(3)}
              </span>
              <span style={{ fontSize: 11, fontFamily: "monospace", color: p.bold ? "#a5b4fc" : "#475569" }}>
                AUC: {p.auc.toFixed(3)}
              </span>
            </div>
          </div>
          <div style={{ height: 5, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                width: `${p.f1 * 100}%`,
                background: p.bold
                  ? "linear-gradient(90deg, #6366f1, #818cf8)"
                  : "#334155",
                borderRadius: 3,
                transition: "width 1s ease",
              }}
            />
          </div>
        </div>
      ))}
      <p style={{ fontSize: 10, color: "#334155", marginTop: 8 }}>
        * Placeholder results. Replace with actual evaluation on held-out test split.
      </p>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function FHIDemo() {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [activeTab, setActiveTab] = useState("overview");
  const [weights, setWeights] = useState({
    w1_aas: 0.30, w2_cis: 0.35, w3_ess: 0.20, w4_hcg: 0.15,
  });

  const sample = DEMO_SAMPLES[selectedIdx];

  const handleWeightChange = useCallback((key, val) => {
    setWeights((prev) => ({ ...prev, [key]: val }));
  }, []);

  const computedFhi = Math.max(
    0,
    Math.min(
      1,
      weights.w1_aas * sample.aas +
        weights.w2_cis * sample.cis +
        weights.w3_ess * sample.ess -
        weights.w4_hcg * sample.hcg
    )
  );

  const isHallucination = computedFhi < 0.5;

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "attribution", label: "Attribution" },
    { id: "perturbation", label: "Perturbation" },
    { id: "baselines", label: "Baselines" },
    { id: "benchmark", label: "Benchmark" },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#020817",
      color: "#e2e8f0",
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
      padding: "24px 16px",
    }}>
      {/* Header */}
      <div style={{ maxWidth: 1100, margin: "0 auto 24px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: "#6366f1",
                boxShadow: "0 0 10px #6366f1",
              }} />
              <span style={{ fontSize: 11, color: "#6366f1", fontWeight: 700, letterSpacing: 2, textTransform: "uppercase" }}>
                Research Demo
              </span>
            </div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: "#f1f5f9", lineHeight: 1.2 }}>
              Faithfulness-Hallucination Index
            </h1>
            <p style={{ margin: "6px 0 0", fontSize: 13, color: "#64748b", maxWidth: 520 }}>
              Multi-dimensional evaluation of explanation faithfulness in LLMs via attribution alignment and causal perturbation analysis
            </p>
          </div>
          {/* FHI verdict badge */}
          <div style={{
            padding: "10px 18px",
            borderRadius: 12,
            background: isHallucination ? "rgba(239,68,68,0.12)" : "rgba(34,197,94,0.12)",
            border: `1px solid ${isHallucination ? "#ef4444" : "#22c55e"}`,
            textAlign: "center",
          }}>
            <div style={{ fontSize: 11, color: isHallucination ? "#f87171" : "#4ade80", fontWeight: 700, letterSpacing: 1 }}>
              {isHallucination ? "⚠ HALLUCINATION" : "✓ FAITHFUL"}
            </div>
            <div style={{ fontSize: 30, fontWeight: 900, color: isHallucination ? "#ef4444" : "#22c55e", fontFamily: "monospace", lineHeight: 1 }}>
              {computedFhi.toFixed(3)}
            </div>
            <div style={{ fontSize: 10, color: "#475569" }}>FHI Score</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", display: "grid", gridTemplateColumns: "220px 1fr", gap: 16 }}>

        {/* ── Left Panel: Sample Selector + Weights ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Sample Selector */}
          <div style={{
            background: "#0a1628",
            border: "1px solid #1e293b",
            borderRadius: 12,
            padding: 14,
          }}>
            <div style={{ fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>
              SAMPLE EXAMPLES
            </div>
            {DEMO_SAMPLES.map((s, i) => (
              <button
                key={s.id}
                onClick={() => setSelectedIdx(i)}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: selectedIdx === i ? "1px solid #6366f1" : "1px solid transparent",
                  background: selectedIdx === i ? "rgba(99,102,241,0.12)" : "transparent",
                  cursor: "pointer",
                  textAlign: "left",
                  marginBottom: 4,
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 700, color: selectedIdx === i ? "#a5b4fc" : "#64748b" }}>
                  {s.tag}
                </div>
                <div style={{
                  fontSize: 10, color: "#334155",
                  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                  maxWidth: 180, marginTop: 2
                }}>
                  {s.question}
                </div>
              </button>
            ))}
          </div>

          {/* FHI Gauge */}
          <div style={{
            background: "#0a1628",
            border: "1px solid #1e293b",
            borderRadius: 12,
            padding: 16,
            display: "flex",
            justifyContent: "center",
          }}>
            <GaugeArc value={computedFhi} label="FHI Score" />
          </div>

          {/* Weight Sliders */}
          <div style={{
            background: "#0a1628",
            border: "1px solid #1e293b",
            borderRadius: 12,
            padding: 14,
          }}>
            <div style={{ fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>
              ADJUST WEIGHTS
            </div>
            <WeightSliders weights={weights} onChangeWeights={handleWeightChange} />
          </div>
        </div>

        {/* ── Right Panel: Main Content ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Sample Info Card */}
          <div style={{
            background: "#0a1628",
            border: "1px solid #1e293b",
            borderRadius: 12,
            padding: 16,
          }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: 10, color: "#475569", fontWeight: 700, marginBottom: 6 }}>QUESTION</div>
                <p style={{ margin: 0, fontSize: 13, color: "#e2e8f0", lineHeight: 1.5 }}>{sample.question}</p>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div style={{ background: "#0f172a", padding: "8px 10px", borderRadius: 8, border: "1px solid #1e293b" }}>
                  <div style={{ fontSize: 9, color: "#475569", marginBottom: 4 }}>MODEL ANSWER</div>
                  <p style={{ margin: 0, fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{sample.answer}</p>
                </div>
                <div style={{
                  background: "#0f172a", padding: "8px 10px", borderRadius: 8,
                  border: `1px solid ${sample.isHallucination ? "#7f1d1d" : "#14532d"}`
                }}>
                  <div style={{ fontSize: 9, color: "#475569", marginBottom: 4 }}>GOLD ANSWER</div>
                  <p style={{ margin: 0, fontSize: 12, color: sample.isHallucination ? "#f87171" : "#4ade80", fontWeight: 600 }}>
                    {sample.goldAnswer}
                  </p>
                </div>
              </div>
            </div>
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid #1e293b" }}>
              <div style={{ fontSize: 10, color: "#475569", fontWeight: 700, marginBottom: 6 }}>MODEL EXPLANATION</div>
              <p style={{ margin: 0, fontSize: 12, color: "#94a3b8", lineHeight: 1.6, fontStyle: "italic" }}>
                "{sample.explanation}"
              </p>
            </div>
          </div>

          {/* Tabs */}
          <div style={{
            background: "#0a1628",
            border: "1px solid #1e293b",
            borderRadius: 12,
            overflow: "hidden",
          }}>
            {/* Tab Bar */}
            <div style={{
              display: "flex",
              borderBottom: "1px solid #1e293b",
              background: "#020817",
            }}>
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    padding: "10px 16px",
                    border: "none",
                    background: activeTab === tab.id ? "#0a1628" : "transparent",
                    color: activeTab === tab.id ? "#a5b4fc" : "#475569",
                    fontWeight: activeTab === tab.id ? 700 : 400,
                    fontSize: 12,
                    cursor: "pointer",
                    borderBottom: activeTab === tab.id ? "2px solid #6366f1" : "2px solid transparent",
                    transition: "all 0.2s",
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div style={{ padding: 16 }}>
              {/* OVERVIEW TAB */}
              {activeTab === "overview" && (
                <div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
                    {[
                      { label: "AAS", val: sample.aas, w: "0.30", desc: "Attribution Alignment", inv: false },
                      { label: "CIS", val: sample.cis, w: "0.35", desc: "Causal Impact", inv: false },
                      { label: "ESS", val: sample.ess, w: "0.20", desc: "Explanation Stability", inv: false },
                      { label: "HCG", val: sample.hcg, w: "0.15", desc: "Confidence Gap", inv: true },
                    ].map((m) => (
                      <div key={m.label} style={{
                        background: "#0f172a",
                        border: `1px solid ${scoreToBg(m.val, m.inv).replace("0.12", "0.5")}`,
                        borderRadius: 10,
                        padding: "12px 14px",
                        textAlign: "center",
                      }}>
                        <div style={{ fontSize: 18, fontWeight: 900, color: scoreToColor(m.val, m.inv), fontFamily: "monospace" }}>
                          {m.val.toFixed(3)}
                        </div>
                        <div style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0", marginTop: 2 }}>{m.label}</div>
                        <div style={{ fontSize: 10, color: "#475569" }}>{m.desc}</div>
                        <div style={{ fontSize: 9, color: "#334155", marginTop: 4 }}>weight: {m.w}</div>
                      </div>
                    ))}
                  </div>

                  <MetricBar label="AAS — Attribution Alignment Score" value={sample.aas} weight="0.30"
                    tooltip="Jaccard overlap between explanation content words and top-K attribution tokens. High = explanation aligns with model attention." />
                  <MetricBar label="CIS — Causal Impact Score" value={sample.cis} weight="0.35"
                    tooltip="Output shift when top-K attribution tokens are masked. High = explanation tokens causally drive the answer." />
                  <MetricBar label="ESS — Explanation Stability Score" value={sample.ess} weight="0.20"
                    tooltip="1 − mean pairwise semantic distance across N=3 explanation runs. High = consistent, grounded reasoning." />
                  <MetricBar label="HCG — Hallucination Confidence Gap" value={sample.hcg} weight="−0.15" inverted={true}
                    tooltip="max(0, confidence − token-F1). High = model is confidently wrong. This is SUBTRACTED from FHI." />

                  {/* FHI formula display */}
                  <div style={{
                    marginTop: 14,
                    padding: "12px 16px",
                    background: "#020817",
                    borderRadius: 10,
                    border: "1px solid #1e293b",
                    fontFamily: "monospace",
                    fontSize: 12,
                    color: "#94a3b8",
                  }}>
                    <span style={{ color: "#6366f1", fontWeight: 700 }}>FHI</span>
                    {" = clip("}
                    <span style={{ color: "#22c55e" }}>{weights.w1_aas.toFixed(2)}</span>{"×"}
                    <span style={{ color: "#4ade80" }}>{sample.aas.toFixed(3)}</span>
                    {" + "}
                    <span style={{ color: "#22c55e" }}>{weights.w2_cis.toFixed(2)}</span>{"×"}
                    <span style={{ color: "#4ade80" }}>{sample.cis.toFixed(3)}</span>
                    {" + "}
                    <span style={{ color: "#22c55e" }}>{weights.w3_ess.toFixed(2)}</span>{"×"}
                    <span style={{ color: "#4ade80" }}>{sample.ess.toFixed(3)}</span>
                    {" − "}
                    <span style={{ color: "#ef4444" }}>{weights.w4_hcg.toFixed(2)}</span>{"×"}
                    <span style={{ color: "#fca5a5" }}>{sample.hcg.toFixed(3)}</span>
                    {") = "}
                    <span style={{ color: scoreToColor(computedFhi), fontWeight: 900, fontSize: 14 }}>
                      {computedFhi.toFixed(3)}
                    </span>
                    <span style={{ color: computedFhi < 0.5 ? "#ef4444" : "#22c55e", marginLeft: 10 }}>
                      {computedFhi < 0.5 ? "< 0.5 → HALLUCINATION" : "≥ 0.5 → FAITHFUL"}
                    </span>
                  </div>
                </div>
              )}

              {/* ATTRIBUTION TAB */}
              {activeTab === "attribution" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  <TokenHeatmap tokens={sample.tokens} title="INPUT TOKEN ATTRIBUTION HEATMAP (Attention Rollout)" />
                  <div style={{ background: "#0f172a", borderRadius: 10, padding: 12, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 10, color: "#64748b", fontWeight: 700, marginBottom: 8 }}>TOP-K ATTRIBUTED TOKENS</div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {[...sample.tokens]
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 5)
                        .map((t, i) => (
                          <div key={i} style={{
                            display: "flex", alignItems: "center", gap: 6,
                            padding: "4px 10px", borderRadius: 20,
                            background: "rgba(99,102,241,0.12)", border: "1px solid #6366f1"
                          }}>
                            <span style={{ fontSize: 11, color: "#a5b4fc", fontWeight: 700 }}>#{i + 1}</span>
                            <span style={{ fontSize: 13, color: "#e2e8f0", fontWeight: 600 }}>{t.text}</span>
                            <span style={{ fontSize: 10, color: "#475569", fontFamily: "monospace" }}>
                              {t.score.toFixed(3)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div style={{ background: "#0f172a", borderRadius: 10, padding: 12, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 10, color: "#64748b", fontWeight: 700, marginBottom: 6 }}>AAS COMPUTATION</div>
                    <p style={{ fontSize: 12, color: "#64748b", margin: 0, lineHeight: 1.6 }}>
                      AAS = |explanation_words ∩ top_k_tokens| / |explanation_words ∪ top_k_tokens|{" "}
                      <span style={{ color: "#a5b4fc", fontFamily: "monospace" }}>
                        = {sample.aas.toFixed(3)}
                      </span>
                    </p>
                    <p style={{ fontSize: 11, color: "#334155", margin: "8px 0 0", lineHeight: 1.5 }}>
                      ⚠️ v2 upgrade: Replace binary Jaccard with gradient-weighted overlap (GAA) to preserve attribution magnitude.
                    </p>
                  </div>
                </div>
              )}

              {/* PERTURBATION TAB */}
              {activeTab === "perturbation" && (
                <div>
                  <div style={{ fontSize: 10, color: "#64748b", fontWeight: 700, marginBottom: 10 }}>
                    CAUSAL PERTURBATION ANALYSIS — MASK STRATEGY
                  </div>
                  <PerturbationView perturbation={sample.perturbation} answer={sample.answer} />
                  <div style={{
                    marginTop: 12,
                    padding: "10px 12px",
                    background: "#0f172a",
                    borderRadius: 8,
                    border: "1px solid #1e293b"
                  }}>
                    <p style={{ fontSize: 11, color: "#475569", margin: 0, lineHeight: 1.6 }}>
                      <span style={{ color: "#f59e0b", fontWeight: 700 }}>⚠️ FHI v1 CIS Issue:</span>{" "}
                      Current implementation masks tokens from the <em>original question</em> — this measures input sensitivity (LIME), not explanation faithfulness.{" "}
                      <span style={{ color: "#4ade80" }}>FHI v2</span> fixes this by testing whether a counterfactual explanation reduces answer confidence.
                    </p>
                  </div>
                </div>
              )}

              {/* BASELINES TAB */}
              {activeTab === "baselines" && (
                <div>
                  <div style={{ fontSize: 10, color: "#64748b", fontWeight: 700, marginBottom: 10 }}>
                    HALLUCINATION DETECTION — THIS SAMPLE
                  </div>
                  <BaselineComparison baselines={sample.baselines} fhi={computedFhi} />
                  <div style={{ marginTop: 12, padding: "10px 12px", background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b" }}>
                    <p style={{ fontSize: 11, color: "#475569", margin: 0, lineHeight: 1.6 }}>
                      Missing baselines to add for publication:{" "}
                      <span style={{ color: "#a5b4fc" }}>SelfCheckGPT (NLI variant)</span>,{" "}
                      <span style={{ color: "#a5b4fc" }}>INSIDE (EigenScore)</span>,{" "}
                      <span style={{ color: "#a5b4fc" }}>SAPLMA (probe classifier)</span>.
                    </p>
                  </div>
                </div>
              )}

              {/* BENCHMARK TAB */}
              {activeTab === "benchmark" && (
                <div>
                  <BenchmarkChart />
                  <div style={{ marginTop: 14, padding: "12px 14px", background: "#0f172a", borderRadius: 10, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 10, color: "#64748b", fontWeight: 700, marginBottom: 8 }}>FHI v2 ROADMAP STATUS</div>
                    {[
                      { label: "Fix CIS: counterfactual explanation perturbation", status: "🔴 Critical", done: false },
                      { label: "Fix JS divergence: vocabulary-level KL", status: "🔴 Critical", done: false },
                      { label: "Fix AAS: subword reconstruction + gradient weighting", status: "🟡 High", done: false },
                      { label: "Add train/val/test split (no data leakage)", status: "🔴 Critical", done: false },
                      { label: "Add McNemar + bootstrap CI significance tests", status: "🟡 High", done: false },
                      { label: "Add SelfCheckGPT, INSIDE baselines", status: "🟡 High", done: false },
                      { label: "ESS v2: NLI-based consistency (N=10)", status: "🟡 High", done: false },
                      { label: "HCG v2: length-normalized confidence", status: "🟡 High", done: false },
                    ].map((item, i) => (
                      <div key={i} style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        padding: "6px 0", borderBottom: i < 7 ? "1px solid #0f172a" : "none"
                      }}>
                        <span style={{ fontSize: 11, color: item.done ? "#4ade80" : "#94a3b8" }}>
                          {item.done ? "✅" : "◻"} {item.label}
                        </span>
                        <span style={{ fontSize: 10, color: "#475569", whiteSpace: "nowrap" }}>{item.status}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ maxWidth: 1100, margin: "16px auto 0", textAlign: "center" }}>
        <p style={{ fontSize: 11, color: "#1e293b" }}>
          FHI = w₁·AAS + w₂·CIS + w₃·ESS − w₄·HCG | Threshold τ = 0.5 | Model: Gemma-2B-IT
        </p>
      </div>
    </div>
  );
}
