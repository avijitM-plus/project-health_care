/**
 * VoiceButton — Premium voice interaction component.
 *
 * Features:
 *   - Animated mic button with state indicators (idle / recording / processing / playing)
 *   - Real-time waveform visualization (canvas-based)
 *   - Recording timer display
 *   - Click to toggle recording
 *   - Visual pulse animation during recording
 */
import React, { useRef, useEffect, useCallback } from 'react';
import { Mic, MicOff, Loader2, Volume2 } from 'lucide-react';
import type { VoiceState } from '../hooks/useVoice';
import './VoiceButton.css';

interface VoiceButtonProps {
    state: VoiceState;
    volume: number;
    frequencyData: Uint8Array | null;
    recordingDuration: number;
    isSupported: boolean;
    error: string | null;
    onToggleRecording: () => void;
    onStopAudio: () => void;
    disabled?: boolean;
}

export const VoiceButton: React.FC<VoiceButtonProps> = ({
    state,
    volume,
    frequencyData,
    recordingDuration,
    isSupported,
    error,
    onToggleRecording,
    onStopAudio,
    disabled = false,
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // ── Waveform visualization ───────────────────────────────────────────

    useEffect(() => {
        if (state !== 'recording' || !frequencyData || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = canvas.offsetWidth * dpr;
        canvas.height = canvas.offsetHeight * dpr;
        ctx.scale(dpr, dpr);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;

        ctx.clearRect(0, 0, width, height);

        // Draw waveform bars
        const barCount = 32;
        const barWidth = width / barCount - 1;
        const step = Math.floor(frequencyData.length / barCount);

        for (let i = 0; i < barCount; i++) {
            const value = frequencyData[i * step] / 255;
            const barHeight = Math.max(2, value * height * 0.8);
            const x = i * (barWidth + 1);
            const y = (height - barHeight) / 2;

            // Gradient from teal to blue based on position
            const hue = 170 + (i / barCount) * 40;
            const lightness = 45 + value * 15;
            ctx.fillStyle = `hsl(${hue}, 80%, ${lightness}%)`;

            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, barHeight, 1);
            ctx.fill();
        }
    }, [state, frequencyData]);

    // ── Format timer ─────────────────────────────────────────────────────

    const formatDuration = (seconds: number): string => {
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    // ── Click handler ────────────────────────────────────────────────────

    const handleClick = useCallback(() => {
        if (state === 'playing') {
            onStopAudio();
        } else if (state !== 'processing') {
            onToggleRecording();
        }
    }, [state, onToggleRecording, onStopAudio]);

    // ── Render ───────────────────────────────────────────────────────────

    const isRecording = state === 'recording';
    const isProcessing = state === 'processing';
    const isPlaying = state === 'playing';

    const buttonClass = [
        'voice-button',
        isRecording && 'voice-button--recording',
        isProcessing && 'voice-button--processing',
        isPlaying && 'voice-button--playing',
        disabled && 'voice-button--disabled',
    ].filter(Boolean).join(' ');

    const getTitle = (): string => {
        if (!isSupported) return 'Voice recording not supported';
        if (isRecording) return 'Click to stop recording';
        if (isProcessing) return 'Processing voice...';
        if (isPlaying) return 'Click to stop playback';
        return 'Click to start voice recording';
    };

    return (
        <div className="voice-button-container">
            {/* Recording waveform visualization */}
            {isRecording && (
                <div className="voice-waveform-container animate-slide-up">
                    <canvas
                        ref={canvasRef}
                        className="voice-waveform-canvas"
                    />
                    <div className="voice-recording-info">
                        <span className="voice-recording-dot" />
                        <span className="voice-recording-timer">
                            {formatDuration(recordingDuration)}
                        </span>
                    </div>
                </div>
            )}

            {/* Processing indicator */}
            {isProcessing && (
                <div className="voice-processing-label animate-slide-up">
                    Processing...
                </div>
            )}

            {/* Main button */}
            <button
                className={buttonClass}
                onClick={handleClick}
                disabled={disabled || !isSupported || isProcessing}
                title={getTitle()}
                aria-label={getTitle()}
                id="voice-record-button"
            >
                {/* Dynamic glow based on volume */}
                {isRecording && (
                    <div
                        className="voice-button-glow"
                        style={{ transform: `scale(${1 + volume * 0.5})` }}
                    />
                )}

                {isProcessing ? (
                    <Loader2 size={20} className="voice-icon-spin" />
                ) : isPlaying ? (
                    <Volume2 size={20} />
                ) : isRecording ? (
                    <MicOff size={20} />
                ) : (
                    <Mic size={20} />
                )}
            </button>

            {/* Error message */}
            {error && (
                <div className="voice-error animate-slide-up">
                    {error}
                </div>
            )}
        </div>
    );
};
