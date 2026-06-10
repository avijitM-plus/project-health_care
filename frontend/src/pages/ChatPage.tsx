import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../context/ChatContext';
import { useLanguage } from '../context/LanguageContext';
import { apiService } from '../services/api';
import { ChatMessage } from '../components/ChatMessage';
import { VoiceButton } from '../components/VoiceButton';
import { useVoice } from '../hooks/useVoice';
import { Send, FileText, RefreshCw, Paperclip, ScanLine, Globe } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import type { VoiceChatResponse } from '../types/api';
import './ChatPage.css';

export const ChatPage: React.FC = () => {
    const { messages, addMessage, conversationId, resetConversation, isTyping, setIsTyping, addImagingStudy, patientAge, patientGender } = useChat();
    const { language, setLanguage, t, voiceId, speed } = useLanguage();
    const [inputValue, setInputValue] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const xrayInputRef = useRef<HTMLInputElement>(null);
    const navigate = useNavigate();

    // ── Language toggle ──────────────────────────────────────────────────
    const toggleLanguage = () => {
        setLanguage(language === 'en' ? 'bn' : 'en');
    };

    // ── Voice system ─────────────────────────────────────────────────────
    const handleVoiceChatComplete = (response: VoiceChatResponse) => {
        // Add user message (transcribed speech)
        addMessage({
            sender: 'user',
            text: response.transcript,
        });

        // Add AI response
        addMessage({
            sender: 'ai',
            text: response.ai_response,
            responseMetadata: {
                reply: response.ai_response,
                urgency: response.urgency,
                followup_questions: response.followup_questions,
                possible_diseases: response.possible_diseases,
                suggested_replies: response.suggested_replies,
                accumulated_symptoms: [],
                predictor_available: true,
                turn_number: 0,
                clinical_slots: {},
                stage: 0,
                advice: '',
                disclaimer: 'This is AI-generated guidance and not a medical diagnosis.',
            },
        });
    };

    const handleVoiceError = (error: string) => {
        addMessage({
            sender: 'ai',
            text: `${t('voice.error')}: ${error}`,
        });
    };

    const voice = useVoice({
        conversationId,
        language,
        silenceThreshold: 15,
        silenceDuration: 2500,
        age: patientAge,
        gender: patientGender,
        voiceId,
        speed,
        onVoiceChatComplete: handleVoiceChatComplete,
        onError: handleVoiceError,
    });

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isTyping]);

    const handleSend = async (text: string = inputValue) => {
        if (!text.trim()) return;

        const userText = text.trim();
        setInputValue('');
        
        // Add user message to UI
        addMessage({
            sender: 'user',
            text: userText
        });

        setIsTyping(true);

        try {
            const response = await apiService.chat({
                message: userText,
                conversation_id: conversationId,
                age: patientAge ?? undefined,
                gender: patientGender ?? undefined,
                language,
            });

            addMessage({
                sender: 'ai',
                text: response.reply,
                responseMetadata: response
            });
        } catch (error) {
            console.error("Chat error:", error);
            addMessage({
                sender: 'ai',
                text: t('chat.error'),
            });
        } finally {
            setIsTyping(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleXRayUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsTyping(true);
        addMessage({
            sender: 'ai',
            text: `Analyzing chest X-ray: ${file.name}...`
        });

        try {
            const response = await apiService.analyzeXray(file, conversationId);

            // Store imaging study in context for sidebar display
            addImagingStudy(response.imaging);

            // Add the auto-generated clinical response as an AI message
            addMessage({
                sender: 'ai',
                text: response.clinical_response,
            });
        } catch (error) {
            console.error('X-ray upload error:', error);
            addMessage({
                sender: 'ai',
                text: 'There was an error analyzing the chest X-ray. Please ensure the image is a valid JPG or PNG file and try again.'
            });
        } finally {
            setIsTyping(false);
            if (xrayInputRef.current) {
                xrayInputRef.current.value = '';
            }
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsTyping(true);
        // Display a temporary system message to indicate upload started
        addMessage({
            sender: 'ai',
            text: `${t('report.analyzing')} ${file.name}...`
        });

        try {
            await apiService.analyzeReport(file, '', conversationId, language);
            
            // Once analysis is done and merged into state, trigger the chat AI automatically
            const response = await apiService.chat({
                message: `[SYSTEM] I have uploaded a medical report (${file.name}). Please review the clinical slots and longitudinal history, provide a brief clinical interpretation, and continue our triage.`,
                conversation_id: conversationId,
                age: patientAge ?? undefined,
                gender: patientGender ?? undefined,
                language,
            });

            addMessage({
                sender: 'ai',
                text: response.reply,
                responseMetadata: response
            });
        } catch (error) {
            console.error("Upload error:", error);
            addMessage({
                sender: 'ai',
                text: t('report.error'),
            });
        } finally {
            setIsTyping(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const isVoiceBusy = voice.state === 'recording' || voice.state === 'processing';

    return (
        <div className="chat-page">
            <header className="chat-header glass-panel">
                <div className="header-actions">
                    <button onClick={resetConversation} className="btn-secondary" title={t('header.new_session')}>
                        <RefreshCw size={18} /> {t('header.new_session')}
                    </button>
                    <button onClick={() => navigate('/report')} className="btn-primary" title={t('header.analyze_report')}>
                        <FileText size={18} /> {t('header.analyze_report')}
                    </button>
                </div>
                <button
                    className="language-toggle-button"
                    onClick={toggleLanguage}
                    title={language === 'en' ? 'Switch to বাংলা' : 'Switch to English'}
                    id="language-toggle"
                >
                    <Globe size={18} />
                    <span className="language-toggle-label">
                        {language === 'en' ? 'EN' : 'বাং'}
                    </span>
                </button>
            </header>

            <main className="chat-container">
                {messages.length === 0 ? (
                    <div className="empty-chat animate-slide-up">
                        <div className="empty-chat-icon">🤖</div>
                        <h2>{t('chat.empty.title')}</h2>
                        <p>{t('chat.empty.subtitle')}</p>
                        <div className="suggestion-chips">
                            <button onClick={() => handleSend(language === 'bn' ? 'আমার জ্বর এবং কাশি হচ্ছে' : 'I have a fever and cough')}>
                                {t('chat.suggestion.fever')}
                            </button>
                            <button onClick={() => handleSend(language === 'bn' ? 'আমার তীব্র বুকে ব্যথা হচ্ছে' : "I'm experiencing severe chest pain")}>
                                {t('chat.suggestion.chest')}
                            </button>
                        </div>
                        <div className="voice-cta">
                            <p className="voice-cta-text">{t('chat.empty.voice_cta')}</p>
                            <VoiceButton
                                state={voice.state}
                                volume={voice.volume}
                                frequencyData={voice.frequencyData}
                                recordingDuration={voice.recordingDuration}
                                isSupported={voice.isSupported}
                                error={voice.error}
                                onToggleRecording={voice.toggleRecording}
                                onStopAudio={voice.stopAudio}
                            />
                        </div>
                    </div>
                ) : (
                    <div className="messages-list">
                        {messages.map(msg => (
                            <ChatMessage 
                                key={msg.id} 
                                message={msg} 
                                onFollowUpSelect={handleSend}
                            />
                        ))}
                        {isTyping && (
                            <div className="typing-indicator animate-slide-up">
                                <span></span><span></span><span></span>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </main>

            <footer className="chat-input-area">
                <div className="input-glass-panel">
                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        accept=".pdf,.png,.jpg,.jpeg"
                        onChange={handleFileUpload}
                    />
                    <input
                        type="file"
                        ref={xrayInputRef}
                        style={{ display: 'none' }}
                        accept=".jpg,.jpeg,.png"
                        onChange={handleXRayUpload}
                    />
                    <button
                        className="upload-inline-button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isTyping || isVoiceBusy}
                        title={t('report.upload_report')}
                    >
                        <Paperclip size={20} />
                    </button>
                    <button
                        className="upload-inline-button xray-upload-button"
                        onClick={() => xrayInputRef.current?.click()}
                        disabled={isTyping || isVoiceBusy}
                        title={t('report.upload_xray')}
                    >
                        <ScanLine size={20} />
                    </button>

                    {/* Voice button — replaces old inline mic */}
                    <VoiceButton
                        state={voice.state}
                        volume={voice.volume}
                        frequencyData={voice.frequencyData}
                        recordingDuration={voice.recordingDuration}
                        isSupported={voice.isSupported}
                        error={voice.error}
                        onToggleRecording={voice.toggleRecording}
                        onStopAudio={voice.stopAudio}
                        disabled={isTyping}
                    />

                    <textarea
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder={t('chat.placeholder')}
                        rows={1}
                        disabled={isTyping || isVoiceBusy}
                    />
                    <button 
                        className="send-button"
                        onClick={() => handleSend()}
                        disabled={!inputValue.trim() || isTyping || isVoiceBusy}
                    >
                        <Send size={20} />
                    </button>
                </div>
            </footer>
        </div>
    );
};
