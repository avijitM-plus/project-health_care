import React, { createContext, useContext, useState } from 'react';
import type { ReactNode } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { ChatResponse, DiseasePrediction, ImagingFindings } from '../types/api';

export interface Message {
    id: string;
    sender: 'user' | 'ai';
    text: string;
    responseMetadata?: ChatResponse; // Store AI rich metadata
    timestamp: Date;
}

interface ChatContextProps {
    messages: Message[];
    addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => void;
    conversationId: string;
    resetConversation: () => void;
    isTyping: boolean;
    setIsTyping: (typing: boolean) => void;
    accumulatedSymptoms: string[];
    latestDiseases: DiseasePrediction[];
    peakUrgency: string;
    clinicalSlots: Record<string, any>;
    stage: number;
    stageName: string;
    progressPercent: number;
    uploadedReports: any[];
    imagingStudies: ImagingFindings[];
    addImagingStudy: (study: ImagingFindings) => void;
    // Patient demographics (explicit user-provided, fed into every API call)
    patientAge: number | null;
    patientGender: string | null;
    setPatientAge: (age: number | null) => void;
    setPatientGender: (gender: string | null) => void;
}

const ChatContext = createContext<ChatContextProps | undefined>(undefined);

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [conversationId, setConversationId] = useState<string>(uuidv4());
    const [isTyping, setIsTyping] = useState<boolean>(false);
    const [imagingStudies, setImagingStudies] = useState<ImagingFindings[]>([]);
    const [patientAge, setPatientAge] = useState<number | null>(null);
    const [patientGender, setPatientGender] = useState<string | null>(null);
    
    // Derived state from latest AI response
    const accumulatedSymptoms = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.accumulated_symptoms)
        .slice(-1)[0]?.responseMetadata?.accumulated_symptoms || [];

    const latestDiseases = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.possible_diseases)
        .slice(-1)[0]?.responseMetadata?.possible_diseases || [];

    const peakUrgency = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.urgency)
        .slice(-1)[0]?.responseMetadata?.urgency || 'NONE';
        
    const clinicalSlots = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.clinical_slots)
        .slice(-1)[0]?.responseMetadata?.clinical_slots || {};

    const stage = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.stage)
        .slice(-1)[0]?.responseMetadata?.stage || 1;

    const stageName = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.stage_name)
        .slice(-1)[0]?.responseMetadata?.stage_name || 'Chief Complaint';

    const progressPercent = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.progress_percent !== undefined)
        .slice(-1)[0]?.responseMetadata?.progress_percent ?? 20;

    const uploadedReports = messages
        .filter(m => m.sender === 'ai' && m.responseMetadata?.reports)
        .slice(-1)[0]?.responseMetadata?.reports || [];

    const addMessage = (msg: Omit<Message, 'id' | 'timestamp'>) => {
        setMessages(prev => [
            ...prev,
            { ...msg, id: uuidv4(), timestamp: new Date() }
        ]);
    };

    const addImagingStudy = (study: ImagingFindings) => {
        setImagingStudies(prev => [...prev, study]);
    };

    const resetConversation = () => {
        setMessages([]);
        setConversationId(uuidv4());
        setImagingStudies([]);
        // Keep age/gender across sessions — patient profile persists
    };

    return (
        <ChatContext.Provider
            value={{
                messages,
                addMessage,
                conversationId,
                resetConversation,
                isTyping,
                setIsTyping,
                accumulatedSymptoms,
                latestDiseases,
                peakUrgency,
                clinicalSlots,
                stage,
                stageName,
                progressPercent,
                uploadedReports,
                imagingStudies,
                addImagingStudy,
                patientAge,
                patientGender,
                setPatientAge,
                setPatientGender,
            }}
        >
            {children}
        </ChatContext.Provider>
    );
};

export const useChat = () => {
    const context = useContext(ChatContext);
    if (!context) {
        throw new Error('useChat must be used within a ChatProvider');
    }
    return context;
};
