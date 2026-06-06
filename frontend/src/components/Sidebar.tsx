import React from 'react';
import { useChat } from '../context/ChatContext';
import { UrgencyBadge } from './UrgencyBadge';
import { Activity, AlertCircle, List, Stethoscope, FileText, ScanLine } from 'lucide-react';
import './Sidebar.css';

export const Sidebar: React.FC = () => {
    const { accumulatedSymptoms, peakUrgency, latestDiseases, clinicalSlots, stage, uploadedReports, imagingStudies } = useChat();
    
    const stageNames = ["Chief Complaint", "Characterization", "Red Flags", "Differential Refinement", "Disposition"];
    const currentStageName = stageNames[Math.min(stage - 1, 4)] || "Consultation";

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <Activity className="brand-icon" size={24} />
                <h1 className="brand-title">IASIS AI</h1>
            </div>

            <div className="sidebar-content">
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
                </section>

                <section className="sidebar-section">
                    <h2 className="section-title">
                        <List size={16} /> Clinical State (Stage {stage}: {currentStageName})
                    </h2>
                    {Object.keys(clinicalSlots).length > 0 ? (
                        <div className="clinical-slots-grid animate-slide-up">
                            {Object.entries(clinicalSlots).map(([key, value], idx) => (
                                <div key={idx} className="slot-item">
                                    <span className="slot-key">{key.replace(/_/g, ' ')}</span>
                                    <span className="slot-value">{String(value)}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">No clinical state extracted yet.</p>
                    )}
                </section>

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
                        <p className="empty-state">Awaiting data...</p>
                    )}
                </section>

                <section className="sidebar-section">
                    <h2 className="section-title">
                        <FileText size={16} /> Report Timeline
                    </h2>
                    {uploadedReports && uploadedReports.length > 0 ? (
                        <div className="reports-timeline animate-slide-up">
                            {uploadedReports.map((r, idx) => (
                                <div key={idx} className="report-mini">
                                    <span className="report-type">📄 {r.report_type}</span>
                                    <span className="report-date">{r.report_date}</span>
                                    {Object.keys(r.findings).length > 0 && (
                                        <span className="report-badge">⚠ Findings Extracted</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">No reports uploaded.</p>
                    )}
                </section>
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <ScanLine size={16} /> Imaging Studies
                    </h2>
                    {imagingStudies && imagingStudies.length > 0 ? (
                        <div className="imaging-timeline animate-slide-up">
                            {imagingStudies.map((study) => (
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
