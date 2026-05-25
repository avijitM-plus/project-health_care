import React from 'react';
import type { DiseasePrediction } from '../types/api';
import './DiseaseCard.css';

interface DiseaseCardProps {
    prediction: DiseasePrediction;
}

export const DiseaseCard: React.FC<DiseaseCardProps> = ({ prediction }) => {
    // Helper to determine CSS class based on concern level
    const getConcernClass = (level: string) => {
        if (level === "High Concern" || level === "Must Rule Out Urgently") return "concern-high";
        if (level === "Moderate Concern") return "concern-moderate";
        return "concern-low";
    };

    const concernClass = getConcernClass(prediction.concern_level);

    return (
        <div className="disease-card animate-slide-up">
            <div className="disease-card-header">
                <span className="disease-name">{prediction.name}</span>
                <span className={`disease-concern-badge ${concernClass}`}>
                    {prediction.concern_level}
                </span>
            </div>
        </div>
    );
};
