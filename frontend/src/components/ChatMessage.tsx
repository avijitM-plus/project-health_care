import React from 'react';
import type { Message } from '../context/ChatContext';
import { UrgencyBadge } from './UrgencyBadge';
import { DiseaseCard } from './DiseaseCard';
import { FollowUpChips } from './FollowUpChips';
import { Bot, User } from 'lucide-react';
import './ChatMessage.css';

interface ChatMessageProps {
    message: Message;
    onFollowUpSelect?: (question: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onFollowUpSelect }) => {
    const isAi = message.sender === 'ai';
    const metadata = message.responseMetadata;

    return (
        <div className={`chat-message-wrapper ${isAi ? 'ai-msg' : 'user-msg'} animate-slide-up`}>
            <div className="chat-avatar">
                {isAi ? <Bot size={20} /> : <User size={20} />}
            </div>
            
            <div className="chat-content">
                <div className="chat-bubble">
                    <p>{message.text}</p>
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
                    </div>
                )}
            </div>
        </div>
    );
};
