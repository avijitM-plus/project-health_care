import React, { useRef, useState } from 'react';
import type { Message } from '../context/ChatContext';
import { UrgencyBadge } from './UrgencyBadge';
import { DiseaseCard } from './DiseaseCard';
import { FollowUpChips } from './FollowUpChips';
import { Bot, User, ThumbsUp, ThumbsDown, Volume2 } from 'lucide-react';
import { apiService } from '../services/api';
import { useChat } from '../context/ChatContext';
import { useLanguage } from '../context/LanguageContext';
import './ChatMessage.css';

interface ChatMessageProps {
    message: Message;
    onFollowUpSelect?: (question: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onFollowUpSelect }) => {
    const isAi = message.sender === 'ai';
    const metadata = message.responseMetadata;
    const { conversationId } = useChat();
    const { language, voiceId, speed } = useLanguage();
    const [feedbackGiven, setFeedbackGiven] = useState<'helpful' | 'incorrect' | null>(null);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    const handleFeedback = async (rating: 'helpful' | 'incorrect') => {
        if (feedbackGiven) return;
        setFeedbackGiven(rating);
        try {
            await apiService.submitFeedback(
                conversationId,
                message.id,
                rating,
                message.text,
            );
        } catch {
            // Feedback is best-effort — don't surface errors to user
        }
    };

    const handleSpeak = async () => {
        if (isSpeaking) {
            audioRef.current?.pause();
            if (audioRef.current?.src) {
                URL.revokeObjectURL(audioRef.current.src);
            }
            audioRef.current = null;
            setIsSpeaking(false);
            return;
        }

        setIsSpeaking(true);
        try {
            const buffer = await apiService.textToSpeech(message.text, voiceId, speed, 'mp3', language);
            const blob = new Blob([buffer], { type: 'audio/mpeg' });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audioRef.current = audio;

            audio.onended = () => {
                URL.revokeObjectURL(url);
                audioRef.current = null;
                setIsSpeaking(false);
            };

            audio.onerror = () => {
                URL.revokeObjectURL(url);
                audioRef.current = null;
                setIsSpeaking(false);
            };

            audio.play();
        } catch (err) {
            console.error('TTS error:', err);
            setIsSpeaking(false);
        }
    };

    return (
        <div className={`chat-message-wrapper ${isAi ? 'ai-msg' : 'user-msg'} animate-slide-up`}>
            <div className="chat-avatar">
                {isAi ? <Bot size={20} /> : <User size={20} />}
            </div>

            <div className="chat-content">
                <div className="chat-bubble">
                    <p>{message.text}</p>
                    {isAi && (
                        <button
                            className={`speak-btn${isSpeaking ? ' speaking' : ''}`}
                            onClick={handleSpeak}
                            title={isSpeaking ? 'Stop speaking' : 'Listen to this message'}
                        >
                            <Volume2 size={13} />
                        </button>
                    )}
                </div>

                {/* AI Metadata Display */}
                {isAi && metadata && (
                    <div className="chat-metadata">
                        {metadata.urgency && metadata.urgency !== 'UNKNOWN' && (
                            <div className="metadata-row">
                                <span className="meta-label">Assessed Urgency:</span>
                                <UrgencyBadge urgency={metadata.urgency} />
                            </div>
                        )}

                        {metadata.possible_diseases && metadata.possible_diseases.length > 0 && (
                            <div className="metadata-section">
                                <span className="meta-label">Possible Conditions:</span>
                                <div className="diseases-grid">
                                    {metadata.possible_diseases.map((d, i) => (
                                        <DiseaseCard key={i} prediction={d} />
                                    ))}
                                </div>
                            </div>
                        )}

                        {metadata.advice && (
                            <div className="advice-box">
                                <strong>Advice: </strong> {metadata.advice}
                            </div>
                        )}

                        {metadata.recommended_tests && metadata.recommended_tests.length > 0 && (
                            <div className="tests-box">
                                <span className="meta-label" style={{color: '#0369a1'}}>Recommended Tests:</span>
                                <ul className="tests-list">
                                    {metadata.recommended_tests.map((t, idx) => (
                                        <li key={idx} className={`test-item ${t.priority?.toLowerCase() === 'immediate' ? 'test-immediate' : 'test-secondary'}`}>
                                            <div className="test-header">
                                                <span className="test-name">{t.test_name}</span>
                                                {t.priority && (
                                                    <span className={`test-priority-badge priority-${t.priority.toLowerCase()}`}>
                                                        {t.priority}
                                                    </span>
                                                )}
                                            </div>
                                            <span className="test-rationale">{t.rationale}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {(metadata.suggested_replies?.length > 0 || metadata.followup_questions?.length > 0) && onFollowUpSelect && (
                            <FollowUpChips
                                questions={metadata.suggested_replies?.length > 0 ? metadata.suggested_replies : metadata.followup_questions}
                                onSelect={onFollowUpSelect}
                            />
                        )}

                        {metadata.disclaimer && (
                            <div className="disclaimer">
                                ⚠️ {metadata.disclaimer}
                            </div>
                        )}

                        <div className="feedback-row">
                            <span className="feedback-label">Was this helpful?</span>
                            <button
                                className={`feedback-btn ${feedbackGiven === 'helpful' ? 'feedback-active-good' : ''}`}
                                onClick={() => handleFeedback('helpful')}
                                disabled={!!feedbackGiven}
                                title="Helpful"
                            >
                                <ThumbsUp size={14} />
                            </button>
                            <button
                                className={`feedback-btn ${feedbackGiven === 'incorrect' ? 'feedback-active-bad' : ''}`}
                                onClick={() => handleFeedback('incorrect')}
                                disabled={!!feedbackGiven}
                                title="Incorrect"
                            >
                                <ThumbsDown size={14} />
                            </button>
                            {feedbackGiven && (
                                <span className="feedback-thanks">Thanks for your feedback</span>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
