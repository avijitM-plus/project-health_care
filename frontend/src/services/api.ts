import type {
    ChatRequest,
    ChatResponse,
    ReportAnalysisResponse,
    XRayAnalysisResponse,
    ClinicalSummary,
    TranscriptionResult,
    VoiceChatResponse,
    VoiceStreamEvent,
    VoiceInfo,
    VoiceHealthResponse,
} from '../types/api';

const API_BASE_URL = '';

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

    async analyzeReport(
        file: File,
        symptoms: string,
        conversationId?: string,
        language?: string,
    ): Promise<ReportAnalysisResponse> {
        const formData = new FormData();
        formData.append('file', file);
        if (symptoms) {
            formData.append('symptoms', symptoms);
        }
        if (conversationId) {
            formData.append('conversation_id', conversationId);
        }
        if (language) {
            formData.append('language', language);
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
    },

    // ── Voice System ─────────────────────────────────────────────────────────

    /**
     * Transcribe audio to text using the STT engine.
     * Passes the user's selected language for accurate recognition.
     */
    async speechToText(
        audioBlob: Blob,
        filename: string = 'recording.webm',
        language: string = 'en',
    ): Promise<TranscriptionResult> {
        const formData = new FormData();
        formData.append('audio', audioBlob, filename);
        formData.append('language', language);

        const response = await fetch(`${API_BASE_URL}/voice/transcribe`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const detail = await response.text().catch(() => response.statusText);
            throw new Error(`Speech-to-text failed: ${detail}`);
        }

        return response.json();
    },

    /**
     * Convert text to speech using the TTS engine.
     * Returns raw audio bytes (ArrayBuffer).
     * Language determines voice selection if no voiceId is provided.
     */
    async textToSpeech(
        text: string,
        voiceId?: string,
        speed: number = 1.0,
        outputFormat: string = 'mp3',
        language: string = 'en',
    ): Promise<ArrayBuffer> {
        const response = await fetch(`${API_BASE_URL}/voice/synthesize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                voice_id: voiceId,
                speed,
                output_format: outputFormat,
                language,
            }),
        });

        if (!response.ok) {
            const detail = await response.text().catch(() => response.statusText);
            throw new Error(`Text-to-speech failed: ${detail}`);
        }

        return response.arrayBuffer();
    },

    /**
     * Full voice chat pipeline — audio → STT → IASIS → TTS → response.
     * Language controls STT recognition, AI response, and TTS voice.
     */
    async voiceChat(
        audioBlob: Blob,
        conversationId: string,
        options?: {
            age?: number;
            gender?: string;
            voiceId?: string;
            speed?: number;
            language?: string;
        },
    ): Promise<VoiceChatResponse> {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('conversation_id', conversationId);

        if (options?.age) formData.append('age', String(options.age));
        if (options?.gender) formData.append('gender', options.gender);
        if (options?.voiceId) formData.append('voice_id', options.voiceId);
        if (options?.speed) formData.append('speed', String(options.speed));
        if (options?.language) formData.append('language', options.language);

        const response = await fetch(`${API_BASE_URL}/voice/chat`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const detail = await response.text().catch(() => response.statusText);
            throw new Error(`Voice chat failed: ${detail}`);
        }

        return response.json();
    },

    /**
     * Streaming voice chat via Server-Sent Events.
     * Calls onEvent for each SSE event (transcript, audio_chunk, clinical, done).
     * Returns an AbortController so the caller can cancel the stream.
     */
    voiceChatStream(
        audioBlob: Blob,
        conversationId: string,
        options: {
            age?: number;
            gender?: string;
            voiceId?: string;
            speed?: number;
            language?: string;
        },
        onEvent: (event: VoiceStreamEvent) => void,
        onError?: (err: Error) => void,
    ): AbortController {
        const controller = new AbortController();

        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('conversation_id', conversationId);
        if (options.age != null) formData.append('age', String(options.age));
        if (options.gender) formData.append('gender', options.gender);
        if (options.voiceId) formData.append('voice_id', options.voiceId);
        formData.append('speed', String(options.speed ?? 1.0));
        formData.append('language', options.language ?? 'en');

        (async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/voice/chat-stream`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal,
                });

                if (!response.ok) {
                    throw new Error(`Voice stream failed: ${response.statusText}`);
                }

                const reader = response.body!.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() ?? '';

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        try {
                            const event: VoiceStreamEvent = JSON.parse(line.slice(6));
                            onEvent(event);
                        } catch {
                            // skip malformed lines
                        }
                    }
                }
            } catch (err: any) {
                if (err?.name !== 'AbortError') {
                    onError?.(err instanceof Error ? err : new Error(String(err)));
                }
            }
        })();

        return controller;
    },

    /**
     * Get available TTS voices.
     */
    async getVoices(): Promise<{ voices: VoiceInfo[] }> {
        const response = await fetch(`${API_BASE_URL}/voice/voices`);
        if (!response.ok) {
            throw new Error(`Failed to fetch voices: ${response.statusText}`);
        }
        return response.json();
    },

    /**
     * Voice system health check.
     */
    async voiceHealthCheck(): Promise<VoiceHealthResponse> {
        const response = await fetch(`${API_BASE_URL}/voice/health`);
        if (!response.ok) {
            throw new Error(`Voice health check failed: ${response.statusText}`);
        }
        return response.json();
    },
};
