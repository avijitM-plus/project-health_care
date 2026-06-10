import React from 'react';
import { useChat } from '../context/ChatContext';
import { useLanguage } from '../context/LanguageContext';
import { UrgencyBadge } from './UrgencyBadge';
import { Activity, AlertCircle, List, Stethoscope, FileText, ScanLine, Pill, User, Settings } from 'lucide-react';
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
        stageName,
        progressPercent,
        uploadedReports,
        imagingStudies,
        patientAge,
        patientGender,
        setPatientAge,
        setPatientGender,
    } = useChat();

    const {
        language,
        setLanguage,
        t,
        voiceGender,
        setVoiceGender,
        speechRate,
        setSpeechRate,
    } = useLanguage();

    const currentStageName = stageName || 'Chief Complaint';

    // Only show meaningfully filled slots
    const filledSlots = Object.entries(clinicalSlots).filter(([, v]) => isFilledSlot(v));

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <Activity className="brand-icon" size={24} />
                <h1 className="brand-title">{t('app.title')}</h1>
            </div>

            <div className="sidebar-content">
                {/* ── Patient Profile ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <User size={16} /> {t('sidebar.patient_profile')}
                    </h2>
                    <div className="profile-form">
                        <div className="profile-field">
                            <label className="profile-label">{t('sidebar.age')}</label>
                            <input
                                type="number"
                                className="profile-input"
                                placeholder={t('sidebar.age_placeholder')}
                                min={1}
                                max={120}
                                value={patientAge ?? ''}
                                onChange={e => setPatientAge(e.target.value ? parseInt(e.target.value, 10) : null)}
                            />
                        </div>
                        <div className="profile-field">
                            <label className="profile-label">{t('sidebar.gender')}</label>
                            <select
                                className="profile-select"
                                value={patientGender ?? ''}
                                onChange={e => setPatientGender(e.target.value || null)}
                            >
                                <option value="">{t('sidebar.gender_not_specified')}</option>
                                <option value="male">{t('sidebar.gender_male')}</option>
                                <option value="female">{t('sidebar.gender_female')}</option>
                                <option value="other">{t('sidebar.gender_other')}</option>
                            </select>
                        </div>
                    </div>
                    {(patientAge || patientGender) && (
                        <p className="profile-hint">
                            {t('sidebar.profile_hint')}
                        </p>
                    )}
                </section>

                {/* ── Status ─────────────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <AlertCircle size={16} /> {t('sidebar.status')}
                    </h2>
                    <div className="status-box">
                        <span className="status-label">{t('sidebar.peak_urgency')}</span>
                        {peakUrgency !== 'NONE' ? (
                            <UrgencyBadge urgency={peakUrgency} />
                        ) : (
                            <span className="status-safe">{t('sidebar.stable')}</span>
                        )}
                    </div>
                    <div className="status-stage">
                        Stage {stage} / 5 — {currentStageName}
                        <div className="stage-bar">
                            <div
                                className="stage-fill"
                                style={{ width: `${progressPercent}%` }}
                            />
                        </div>
                    </div>
                </section>

                {/* ── Reported Symptoms ───────────────────────────────── */}
                {accumulatedSymptoms.length > 0 && (
                    <section className="sidebar-section">
                        <h2 className="section-title">
                            <Pill size={16} /> {t('sidebar.symptoms')}
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
                        <List size={16} /> {t('sidebar.clinical_state')}
                        {filledSlots.length > 0 && (
                            <span className="count-badge">{filledSlots.length} {t('sidebar.filled')}</span>
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
                        <p className="empty-state">{t('sidebar.no_clinical_data')}</p>
                    )}
                </section>

                {/* ── Top Predictions ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <Stethoscope size={16} /> {t('sidebar.predictions')}
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
                        <p className="empty-state">{t('sidebar.awaiting')}</p>
                    )}
                </section>

                {/* ── Report Timeline ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <FileText size={16} /> {t('sidebar.reports')}
                        {uploadedReports.length > 0 && (
                            <span className="count-badge">{uploadedReports.length}</span>
                        )}
                    </h2>
                    {uploadedReports.length > 0 ? (
                        <div className="reports-timeline animate-slide-up">
                            {uploadedReports.map((r, idx) => (
                                <div key={idx} className="report-mini">
                                    <span className="report-type">📄 {r.report_type}</span>
                                    <span className="report-date">{r.report_date || t('sidebar.unknown_date')}</span>
                                    {r.findings && Object.keys(r.findings).length > 0 && (
                                        <span className="report-badge">{t('sidebar.findings_extracted')}</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">{t('sidebar.no_reports')}</p>
                    )}
                </section>

                {/* ── Imaging Studies ─────────────────────────────────── */}
                <section className="sidebar-section">
                    <h2 className="section-title">
                        <ScanLine size={16} /> {t('sidebar.imaging')}
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
                                        <span className="xray-badge-normal">{t('sidebar.no_abnormalities')}</span>
                                    )}
                                    <p className="xray-impression">{study.impression}</p>
                                    <span className="xray-confidence">
                                        {t('sidebar.confidence')}: {Math.round(study.confidence * 100)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="empty-state">{t('sidebar.no_imaging')}</p>
                    )}
                </section>

                {/* ── Language & Voice Settings ────────────────────────── */}
                <section className="sidebar-section settings-section">
                    <h2 className="section-title">
                        <Settings size={16} /> {t('settings.title')}
                    </h2>

                    {/* Language */}
                    <div className="settings-group">
                        <label className="settings-label">{t('settings.language')}</label>
                        <div className="settings-toggle-group">
                            <button
                                className={`settings-toggle-btn ${language === 'en' ? 'active' : ''}`}
                                onClick={() => setLanguage('en')}
                            >
                                English
                            </button>
                            <button
                                className={`settings-toggle-btn ${language === 'bn' ? 'active' : ''}`}
                                onClick={() => setLanguage('bn')}
                            >
                                বাংলা
                            </button>
                        </div>
                    </div>

                    {/* Voice Gender */}
                    <div className="settings-group">
                        <label className="settings-label">{t('settings.voice')}</label>
                        <div className="settings-toggle-group">
                            <button
                                className={`settings-toggle-btn ${voiceGender === 'female' ? 'active' : ''}`}
                                onClick={() => setVoiceGender('female')}
                            >
                                {t('settings.voice_female')}
                            </button>
                            <button
                                className={`settings-toggle-btn ${voiceGender === 'male' ? 'active' : ''}`}
                                onClick={() => setVoiceGender('male')}
                            >
                                {t('settings.voice_male')}
                            </button>
                        </div>
                    </div>

                    {/* Speech Rate */}
                    <div className="settings-group">
                        <label className="settings-label">{t('settings.speech_rate')}</label>
                        <div className="settings-toggle-group settings-toggle-triple">
                            <button
                                className={`settings-toggle-btn ${speechRate === 'slow' ? 'active' : ''}`}
                                onClick={() => setSpeechRate('slow')}
                            >
                                {t('settings.speed_slow')}
                            </button>
                            <button
                                className={`settings-toggle-btn ${speechRate === 'normal' ? 'active' : ''}`}
                                onClick={() => setSpeechRate('normal')}
                            >
                                {t('settings.speed_normal')}
                            </button>
                            <button
                                className={`settings-toggle-btn ${speechRate === 'fast' ? 'active' : ''}`}
                                onClick={() => setSpeechRate('fast')}
                            >
                                {t('settings.speed_fast')}
                            </button>
                        </div>
                    </div>
                </section>
            </div>

            <div className="sidebar-footer">
                <p>{t('app.subtitle')}</p>
            </div>
        </aside>
    );
};
