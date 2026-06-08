export interface DiseasePrediction {
    name: string;
    concern_level: string;
}

export interface ChatResponse {
    reply: string;
    possible_diseases: DiseasePrediction[];
    urgency: string;
    followup_questions: string[];
    advice: string;
    disclaimer: string;
    accumulated_symptoms: string[];
    predictor_available: boolean;
    turn_number: number;
    clinical_slots: Record<string, any>;
    stage: number;
    suggested_replies: string[];
    recommended_tests?: { test_name: string; priority?: string; rationale: string; }[];
    reports?: any[];
    preferred_language?: string;
}

export interface ChatRequest {
    message: string;
    conversation_id: string;
    age?: number | null;
    gender?: string | null;
    chronic_conditions?: string | null;
}

export interface ClinicalSummary {
    chief_complaint: string;
    symptoms: string[];
    clinical_findings: Record<string, any>;
    uploaded_reports: { type: string; date: string; key_findings: string }[];
    imaging_studies: { filename: string; impression: string; abnormalities: string[] }[];
    peak_urgency: string;
    possible_conditions: string[];
    recommended_tests: string[];
    recommended_next_steps: string;
    conversation_turns: number;
    disclaimer: string;
}

export interface ImagingFindings {
    study_id: string;
    modality: string;
    findings: string[];
    abnormalities: string[];
    impression: string;
    confidence: number;
    urgency_hint: string;
    filename: string;
    uploaded_at: number;
}

export interface XRayAnalysisResponse {
    imaging: ImagingFindings;
    clinical_response: string;
    followup_questions: string[];
    urgency: string;
    updated_slots: Record<string, any>;
    disclaimer: string;
}

export interface ReportAnalysisResponse {
    filename: string;
    extracted_text: string;
    analysis: {
        report_date?: string;
        report_type?: string;
        findings?: Record<string, any>;
        summary: string;
        clinical_slots?: Record<string, boolean>;
        extracted_symptoms?: string[];
        possible_conditions: string[];
        advice: string;
        disclaimer: string;
        trend_summary?: string;
    };
}


// ── Voice system types ──────────────────────────────────────────────────────

export interface TranscriptionSegment {
    text: string;
    start: number;
    end: number;
    confidence: number;
}

export interface TranscriptionResult {
    transcript: string;
    language: string;
    confidence: number;
    processing_time: number;
    segments: TranscriptionSegment[];
    engine: string;
}

export interface VoiceChatResponse {
    // STT result
    transcript: string;
    language: string;
    stt_confidence: number;
    // Clinical engine response
    ai_response: string;
    urgency: string;
    followup_questions: string[];
    possible_diseases: DiseasePrediction[];
    suggested_replies: string[];
    // TTS audio (base64-encoded)
    audio_base64: string;
    audio_format: string;
    preferred_language?: string;
    // Performance
    stt_time: number;
    llm_time: number;
    tts_time: number;
    total_time: number;
}

export interface VoiceInfo {
    voice_id: string;
    name: string;
    gender: string;
    language: string;
    engine: string;
    is_default: boolean;
}

export interface VoiceHealthResponse {
    stt_engine: string;
    stt_ready: boolean;
    stt_device: string;
    tts_engine: string;
    tts_ready: boolean;
    tts_voices: number;
}
