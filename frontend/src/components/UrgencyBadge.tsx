import React from 'react';
import './UrgencyBadge.css';

interface UrgencyBadgeProps {
    urgency: string;
}

export const UrgencyBadge: React.FC<UrgencyBadgeProps> = ({ urgency }) => {
    const getUrgencyClass = (level: string) => {
        switch (level?.toUpperCase()) {
            case 'EMERGENCY': return 'urgency-emergency';
            case 'HIGH': return 'urgency-high';
            case 'MEDIUM': return 'urgency-medium';
            case 'LOW': return 'urgency-low';
            default: return 'urgency-none';
        }
    };

    if (!urgency || urgency.toUpperCase() === 'UNKNOWN') return null;

    return (
        <span className={`urgency-badge ${getUrgencyClass(urgency)}`}>
            {urgency.toUpperCase()}
        </span>
    );
};
