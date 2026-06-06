import type { ChatRequest, ChatResponse, ReportAnalysisResponse, XRayAnalysisResponse, ClinicalSummary } from '../types/api';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const apiService = {
    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            return response.ok;
        } catch (error) {
            console.error("Health check failed:", error);
            return false;
        }
    },

    async chat(request: ChatRequest): Promise<ChatResponse> {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`Chat API failed: ${response.statusText}`);
        }

        return response.json();
    },

    async analyzeReport(file: File, symptoms: string, conversationId?: string): Promise<ReportAnalysisResponse> {
        const formData = new FormData();
        formData.append('file', file);
        if (symptoms) {
            formData.append('symptoms', symptoms);
        }
        if (conversationId) {
            formData.append('conversation_id', conversationId);
        }

        const response = await fetch(`${API_BASE_URL}/analyze-report`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Report analysis failed: ${response.statusText}`);
        }

        return response.json();
    },

    async getClinicalSummary(conversationId: string): Promise<ClinicalSummary> {
        const response = await fetch(`${API_BASE_URL}/session/${conversationId}/summary`);
        if (!response.ok) {
            throw new Error(`Clinical summary failed: ${response.statusText}`);
        }
        return response.json();
    },

    async submitFeedback(
        conversationId: string,
        messageId: string,
        rating: 'helpful' | 'incorrect',
        aiReplyExcerpt?: string,
    ): Promise<void> {
        await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: conversationId,
                message_id: messageId,
                rating,
                ai_reply_excerpt: aiReplyExcerpt?.slice(0, 200),
            }),
        });
    },

    async analyzeXray(file: File, conversationId?: string): Promise<XRayAnalysisResponse> {
        const formData = new FormData();
        formData.append('file', file);
        if (conversationId) {
            formData.append('conversation_id', conversationId);
        }

        const response = await fetch(`${API_BASE_URL}/analyze-xray`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`X-ray analysis failed: ${response.statusText}`);
        }

        return response.json();
    }
};
