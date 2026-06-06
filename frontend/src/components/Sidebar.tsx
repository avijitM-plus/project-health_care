import React from 'react';
import { useChat } from '../context/ChatContext';
import { UrgencyBadge } from './UrgencyBadge';
import { Activity, AlertCircle, List, Stethoscope, FileText, ScanLine, Pill, User } from 'lucide-react';
import './Sidebar.css';

// Format slot key: "cough_sputum_blood" → "Cough Sputum Blood"
function formatSlotKey(key: string): string {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Format slot value: boolean → Yes/No, long strings truncated
function formatSlotValue(value: unknown): string {
    if (value === true) return 'Yes';
    if (value === false) return 'No';
    const s = String(value);
    return s.length > 22 ? s.slice(0, 20) + '…' : s;
}

// Returns true if a slot value is meaningful (not empty/unknown/null)
function isFilledSlot(value: unknown): boolean {
    if (value === null || value === undefined) return false;
    if (value === false) return false; // false = negative answer, not useful to display
    if (typeof value === 'string' && ['', 'UNKNOWN', 'unknown', 'null', 'None'].includes(value)) return false;
    return true;
}

export const Sidebar: React.FC = () => {
    const {
        accumulatedSymptoms,
        peakUrgency,
        latestDiseases,
        clinicalSlots,
        stage,
        uploadedReports,
        imagingStudies,
        patientAge,
        patientGender,
        setPatientAge,
        setPatientGender,
    } = useChat();

    const stageNames = [
        'Chief Complaint',
        'Characterization',
        'Red Flags',
        'Differential Refinement',
        'Disposition',
    ];
    const currentStageName = stageNames[Math.min(stage - 1, 4)] || 'Consultation';

    // Only show meaningfully filled slots
    const filledSlots = Object.entries(clinicalSlots).filter(([, v]) => isFilledSlot(v));

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <Activity className="brand-icon" size={24} />
                <h1 className="brand-title">IASIS AI</h1>
            </div>

            <div className="sidebar-content">
                {/* ── Patient Profile ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <User size={16} /> Patient Profile
                    </h2>
                    <div className="profile-form">
                        <div className="profile-field">
                            <label className="profile-label">Age</label>
                            <input
                                type="number"
                                className="profile-input"
                                placeholder="e.g. 35"
                                min={1}
                                max={120}
                                value={patientAge ?? ''}
                                onChange={e => setPatientAge(e.target.value ? parseInt(e.target.value, 10) : null)}
                            />
                        </div>
                        <div className="profile-field">
                            <label className="profile-label">Gender</label>
                            <select
                                className="profile-select"
                                value={patientGender ?? ''}
                                onChange={e => setPatientGender(e.target.value || null)}
                            >
                                <option value="">Not specified</option>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                    </div>
                    {(patientAge || patientGender) && (
                        <p className="profile-hint">
                            Profile sent with every message for accurate predictions.
                        </p>
                    )}
                </section>

                {/* ── Status ─────────────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <AlertCircle size={16} /> Status
                    </h2>
                    <div className="status-box">
                        <span className="status-label">Peak Urgency:</span>
                        {peakUrgency !== 'NONE' ? (
                            <UrgencyBadge urgency={peakUrgency} />
                        ) : (
                            <span className="status-safe">Stable</span>
                        )}
                    </div>
                    <div className="status-stage">
                        Stage {stage} / 5 — {currentStageName}
                        <div className="stage-bar">
                            <div
                                className="stage-fill"
                                style={{ width: `${(stage / 5) * 100}%` }}
                            />
                        </div>
                    </div>
                </section>

                {/* ── Reported Symptoms ───────────────────────────────── */}
                {accumulatedSymptoms.length > 0 && (
                    <section className="sidebar-section">
                        <h2 className="section-title">
                            <Pill size={16} /> Reported Symptoms
                            <span className="count-badge">{accumulatedSymptoms.length}</span>
                        </h2>
                        <div className="symptoms-wrap animate-slide-up">
                            {accumulatedSymptoms.map((sym, i) => (
                                <span key={i} className="symptom-chip">
                                    {sym.replace(/_/g, ' ')}
                                </span>
                            ))}
                        </div>
                    </section>
                )}

                {/* ── Clinical State ──────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <List size={16} /> Clinical State
                        {filledSlots.length > 0 && (
                            <span className="count-badge">{filledSlots.length} filled</span>
                        )}
                    </h2>
                    {filledSlots.length > 0 ? (
                        <div className="clinical-slots-grid animate-slide-up">
                            {filledSlots.map(([key, value]) => (
                                <div key={key} className="slot-item">
                                    <span className="slot-key">{formatSlotKey(key)}</span>
                                    <span className="slot-value">{formatSlotValue(value)}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">No clinical data extracted yet.</p>
                    )}
                </section>

                {/* ── Top Predictions ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <Stethoscope size={16} /> Top Predictions
                    </h2>
                    {latestDiseases.length > 0 ? (
                        <div className="predictions-list animate-slide-up">
                            {latestDiseases.slice(0, 3).map((d, idx) => (
                                <div key={idx} className="prediction-mini">
                                    <span className="pred-name">{d.name}</span>
                                    <span className="pred-prob">{d.concern_level}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">Awaiting data…</p>
                    )}
                </section>

                {/* ── Report Timeline ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <FileText size={16} /> Report Timeline
                        {uploadedReports.length > 0 && (
                            <span className="count-badge">{uploadedReports.length}</span>
                        )}
                    </h2>
                    {uploadedReports.length > 0 ? (
                        <div className="reports-timeline animate-slide-up">
                            {uploadedReports.map((r, idx) => (
                                <div key={idx} className="report-mini">
                                    <span className="report-type">📄 {r.report_type}</span>
                                    <span className="report-date">{r.report_date || 'Unknown date'}</span>
                                    {r.findings && Object.keys(r.findings).length > 0 && (
                                        <span className="report-badge">⚠ Findings extracted</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">No reports uploaded.</p>
                    )}
                </section>

                {/* ── Imaging Studies ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <ScanLine size={16} /> Imaging Studies
                        {imagingStudies.length > 0 && (
                            <span className="count-badge">{imagingStudies.length}</span>
                        )}
                    </h2>
                    {imagingStudies.length > 0 ? (
                        <div className="imaging-timeline animate-slide-up">
                            {imagingStudies.map(study => (
                                <div
                                    key={study.study_id}
                                    className={`xray-mini ${study.abnormalities.length > 0 ? 'xray-abnormal' : ''}`}
                                >
                                    <div className="xray-header">
                                        <span className="xray-filename" title={study.filename}>
                                            {study.filename.length > 22
                                                ? study.filename.slice(0, 20) + '…'
                                                : study.filename}
                                        </span>
                                        <span className="xray-modality-badge">X-Ray</span>
                                    </div>
                                    {study.abnormalities.length > 0 ? (
                                        <span className="xray-badge-abnormal">
                                            ⚠ {study.abnormalities.length} finding{study.abnormalities.length > 1 ? 's' : ''}
                                        </span>
                                    ) : (
                                        <span className="xray-badge-normal">✓ No abnormalities</span>
                                    )}
                                    <p className="xray-impression">{study.impression}</p>
                                    <span className="xray-confidence">
                                        Confidence: {Math.round(study.confidence * 100)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">No imaging studies uploaded.</p>
                    )}
                </section>
            </div>

            <div className="sidebar-footer">
                <p>Medical Triage AI v2.0</p>
            </div>
        </aside>
    );
};
