import React from 'react';
import './FollowUpChips.css';

interface FollowUpChipsProps {
    questions: string[];
    onSelect: (question: string) => void;
}

export const FollowUpChips: React.FC<FollowUpChipsProps> = ({ questions, onSelect }) => {
    if (!questions || questions.length === 0) return null;

    return (
        <div className="followup-container animate-slide-up">
            <p className="followup-title">Suggested replies:</p>
            <div className="followup-chips">
                {questions.map((q, idx) => (
                    <button 
                        key={idx} 
                        className="followup-chip"
                        onClick={() => onSelect(q)}
                    >
                        {q}
                    </button>
                ))}
            </div>
        </div>
    );
};
