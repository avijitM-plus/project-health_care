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
}

export interface ChatRequest {
    message: string;
    conversation_id: string;
    age?: number | null;
    gender?: string | null;
    chronic_conditions?: string | null;
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
