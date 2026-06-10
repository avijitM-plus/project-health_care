import React, { useState } from 'react';
import { apiService } from '../services/api';
import type { ReportAnalysisResponse } from '../types/api';
import { UploadCloud, ArrowLeft, Loader2, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useChat } from '../context/ChatContext';
import { useLanguage } from '../context/LanguageContext';
import './ReportPage.css';

export const ReportPage: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [symptoms, setSymptoms] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<ReportAnalysisResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const { conversationId } = useChat();
    const { language, t } = useLanguage();
    
    const navigate = useNavigate();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError(t('report.error'));
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            // Note: OCR extraction remains language-independent.
            // The language is passed here to ensure the AI *explanation* of the report
            // is generated in the user's selected language.
            const data = await apiService.analyzeReport(file, symptoms, conversationId, language);
            setResult(data);
        } catch (err) {
            console.error(err);
            setError(t('report.error'));
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="report-page">
            <header className="report-header glass-panel">
                <button onClick={() => navigate('/')} className="btn-secondary">
                    <ArrowLeft size={18} /> {t('app.title')}
                </button>
                <h2>{t('report.title')}</h2>
            </header>

            <main className="report-container">
                <div className="upload-section glass-panel">
                    <h3>{t('report.upload')}</h3>
                    
                    <div className="file-drop-area">
                        <input 
                            type="file" 
                            accept=".pdf,.png,.jpg,.jpeg" 
                            onChange={handleFileChange}
                            id="file-upload"
                            className="file-input"
                        />
                        <label htmlFor="file-upload" className="file-label">
                            <UploadCloud size={48} className="upload-icon" />
                            <span className="upload-text">
                                {file ? file.name : t('report.upload_report')}
                            </span>
                        </label>
                    </div>

                    <div className="symptoms-input">
                        <textarea
                            id="symptoms"
                            value={symptoms}
                            onChange={(e) => setSymptoms(e.target.value)}
                            placeholder={t('chat.placeholder')}
                            rows={3}
                        />
                    </div>

                    {error && <div className="error-message"><AlertTriangle size={16}/> {error}</div>}

                    <button 
                        className="btn-primary upload-btn" 
                        onClick={handleUpload}
                        disabled={!file || isLoading}
                    >
                        {isLoading ? <><Loader2 size={18} className="spin" /> {t('report.analyzing')}</> : t('report.upload')}
                    </button>
                </div>

                {result && (
                    <div className="result-section glass-panel animate-slide-up">
                        <h3>Analysis Results</h3>
                        
                        <div className="result-block">
                            <h4>Summary</h4>
                            <p>{result.analysis?.summary || 'No summary available.'}</p>
                        </div>

                        {result.analysis?.trend_summary && (
                            <div className="result-block highlight-block">
                                <h4>📈 Longitudinal Trends</h4>
                                <p>{result.analysis.trend_summary}</p>
                            </div>
                        )}

                        {result.analysis?.findings && Object.keys(result.analysis.findings).length > 0 && (
                            <div className="result-block">
                                <h4>Key Findings</h4>
                                <div className="key-values-grid">
                                    {Object.entries(result.analysis.findings).map(([key, val], idx) => (
                                        <div key={idx} className="kv-item">
                                            <span className="kv-key">{key}</span>
                                            <span className="kv-val">{typeof val === 'object' ? JSON.stringify(val) : String(val)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {result.analysis?.possible_conditions && Array.isArray(result.analysis.possible_conditions) && result.analysis.possible_conditions.length > 0 && (
                            <div className="result-block">
                                <h4>Possible Conditions</h4>
                                <ul className="conditions-list">
                                    {result.analysis.possible_conditions.map((c, i) => (
                                        <li key={i}>{typeof c === 'string' ? c : JSON.stringify(c)}</li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        <div className="result-block advice-block">
                            <h4>Advice</h4>
                            <p>{result.analysis?.advice || 'Please consult a doctor.'}</p>
                        </div>

                        <div className="result-block disclaimer-block">
                            <p>⚠️ {result.analysis?.disclaimer || 'This is AI-generated guidance and not a medical diagnosis. Consult a licensed doctor for professional medical advice.'}</p>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};
