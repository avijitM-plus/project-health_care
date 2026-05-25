import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../context/ChatContext';
import { apiService } from '../services/api';
import { ChatMessage } from '../components/ChatMessage';
import { Send, FileText, RefreshCw, Paperclip } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './ChatPage.css';

export const ChatPage: React.FC = () => {
    const { messages, addMessage, conversationId, resetConversation, isTyping, setIsTyping } = useChat();
    const [inputValue, setInputValue] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const navigate = useNavigate();

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
                conversation_id: conversationId
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
                text: "I'm sorry, I encountered an error connecting to the server. Please try again."
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

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsTyping(true);
        // Display a temporary system message to indicate upload started
        addMessage({
            sender: 'ai',
            text: `Analyzing uploaded report: ${file.name}...`
        });

        try {
            await apiService.analyzeReport(file, '', conversationId);
            
            // Once analysis is done and merged into state, trigger the chat AI automatically
            const response = await apiService.chat({
                message: `[SYSTEM] I have uploaded a medical report (${file.name}). Please review the clinical slots and longitudinal history, provide a brief clinical interpretation, and continue our triage.`,
                conversation_id: conversationId
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
                text: "There was an error analyzing the report. Please try again."
            });
        } finally {
            setIsTyping(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    return (
        <div className="chat-page">
            <header className="chat-header glass-panel">
                <div className="header-actions">
                    <button onClick={resetConversation} className="btn-secondary" title="New Chat">
                        <RefreshCw size={18} /> New Session
                    </button>
                    <button onClick={() => navigate('/report')} className="btn-primary" title="Upload Report">
                        <FileText size={18} /> Analyze Report
                    </button>
                </div>
            </header>

            <main className="chat-container">
                {messages.length === 0 ? (
                    <div className="empty-chat animate-slide-up">
                        <div className="empty-chat-icon">🤖</div>
                        <h2>How can I help you today?</h2>
                        <p>Describe your symptoms, and I will assist in determining the possible causes and urgency.</p>
                        <div className="suggestion-chips">
                            <button onClick={() => handleSend("I have a fever and cough")}>"I have a fever and cough"</button>
                            <button onClick={() => handleSend("I'm experiencing severe chest pain")}>"I'm experiencing severe chest pain"</button>
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
                    <button 
                        className="upload-inline-button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isTyping}
                        title="Upload Medical Report"
                    >
                        <Paperclip size={20} />
                    </button>
                    <textarea
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder="Describe your symptoms here..."
                        rows={1}
                        disabled={isTyping}
                    />
                    <button 
                        className="send-button"
                        onClick={() => handleSend()}
                        disabled={!inputValue.trim() || isTyping}
                    >
                        <Send size={20} />
                    </button>
                </div>
            </footer>
        </div>
    );
};
