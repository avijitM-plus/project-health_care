/**
 * i18n — Static translation dictionary for IASIS AI.
 *
 * Supports English (en) and Bangla (bn).
 * Used by the LanguageContext `t()` helper to translate UI strings.
 */

export type SupportedLanguage = 'en' | 'bn';

const translations: Record<SupportedLanguage, Record<string, string>> = {
  en: {
    // ── App / Header ──────────────────────────────────────────────
    'app.title': 'IASIS AI',
    'app.subtitle': 'Medical Triage AI v2.0',
    'header.new_session': 'New Session',
    'header.analyze_report': 'Analyze Report',
    'header.lang_label': 'EN',

    // ── Chat Page ─────────────────────────────────────────────────
    'chat.placeholder': 'Describe your symptoms here...',
    'chat.empty.title': 'How can I help you today?',
    'chat.empty.subtitle': 'Describe your symptoms, and I will assist in determining the possible causes and urgency.',
    'chat.empty.voice_cta': 'Or speak your symptoms',
    'chat.suggestion.fever': '"I have a fever and cough"',
    'chat.suggestion.chest': '"I\'m experiencing severe chest pain"',
    'chat.typing': 'Thinking...',
    'chat.processing': 'Processing...',
    'chat.error': "I'm sorry, I encountered an error connecting to the server. Please try again.",

    // ── Voice ─────────────────────────────────────────────────────
    'voice.not_supported': 'Voice recording not supported',
    'voice.start': 'Click to start voice recording',
    'voice.stop': 'Click to stop recording',
    'voice.processing': 'Processing voice...',
    'voice.playing': 'Click to stop playback',
    'voice.error': 'Voice error',
    'voice.mic_denied': 'Microphone permission denied. Please allow access in browser settings.',
    'voice.mic_not_found': 'No microphone found. Please connect a microphone.',

    // ── Sidebar ───────────────────────────────────────────────────
    'sidebar.patient_profile': 'Patient Profile',
    'sidebar.age': 'Age',
    'sidebar.age_placeholder': 'e.g. 35',
    'sidebar.gender': 'Gender',
    'sidebar.gender_not_specified': 'Not specified',
    'sidebar.gender_male': 'Male',
    'sidebar.gender_female': 'Female',
    'sidebar.gender_other': 'Other',
    'sidebar.profile_hint': 'Profile sent with every message for accurate predictions.',
    'sidebar.status': 'Status',
    'sidebar.peak_urgency': 'Peak Urgency:',
    'sidebar.stable': 'Stable',
    'sidebar.symptoms': 'Reported Symptoms',
    'sidebar.clinical_state': 'Clinical State',
    'sidebar.filled': 'filled',
    'sidebar.no_clinical_data': 'No clinical data extracted yet.',
    'sidebar.predictions': 'Top Predictions',
    'sidebar.awaiting': 'Awaiting data…',
    'sidebar.reports': 'Report Timeline',
    'sidebar.no_reports': 'No reports uploaded.',
    'sidebar.findings_extracted': '⚠ Findings extracted',
    'sidebar.unknown_date': 'Unknown date',
    'sidebar.imaging': 'Imaging Studies',
    'sidebar.no_imaging': 'No imaging studies uploaded.',
    'sidebar.no_abnormalities': '✓ No abnormalities',
    'sidebar.confidence': 'Confidence',

    // ── Settings ──────────────────────────────────────────────────
    'settings.title': 'Settings',
    'settings.language': 'Language',
    'settings.voice': 'Voice',
    'settings.voice_male': 'Male',
    'settings.voice_female': 'Female',
    'settings.speech_rate': 'Speech Rate',
    'settings.speed_slow': 'Slow',
    'settings.speed_normal': 'Normal',
    'settings.speed_fast': 'Fast',

    // ── Report Page ───────────────────────────────────────────────
    'report.title': 'Medical Report Analysis',
    'report.upload': 'Upload Report',
    'report.analyzing': 'Analyzing report...',
    'report.error': 'There was an error analyzing the report. Please try again.',
    'report.upload_xray': 'Upload Chest X-Ray (JPG/PNG) — analyzed by MedGemma AI',
    'report.upload_report': 'Upload Medical Report (PDF or image)',

    // ── Stage names ───────────────────────────────────────────────
    'stage.1': 'Chief Complaint',
    'stage.2': 'Symptom Exploration',
    'stage.3': 'Risk Assessment',
    'stage.4': 'Report Analysis',
    'stage.5': 'Clinical Recommendation',
  },

  bn: {
    // ── App / Header ──────────────────────────────────────────────
    'app.title': 'IASIS AI',
    'app.subtitle': 'মেডিকেল ট্রায়াজ AI v2.0',
    'header.new_session': 'নতুন সেশন',
    'header.analyze_report': 'রিপোর্ট বিশ্লেষণ',
    'header.lang_label': 'বাং',

    // ── Chat Page ─────────────────────────────────────────────────
    'chat.placeholder': 'আপনার লক্ষণগুলি এখানে বর্ণনা করুন...',
    'chat.empty.title': 'আজ আপনাকে কীভাবে সাহায্য করতে পারি?',
    'chat.empty.subtitle': 'আপনার লক্ষণগুলি বর্ণনা করুন, আমি সম্ভাব্য কারণ এবং জরুরিতা নির্ণয়ে সাহায্য করব।',
    'chat.empty.voice_cta': 'অথবা আপনার লক্ষণগুলি বলুন',
    'chat.suggestion.fever': '"আমার জ্বর এবং কাশি হচ্ছে"',
    'chat.suggestion.chest': '"আমার তীব্র বুকে ব্যথা হচ্ছে"',
    'chat.typing': 'চিন্তা করছি...',
    'chat.processing': 'প্রক্রিয়াকরণ হচ্ছে...',
    'chat.error': 'দুঃখিত, সার্ভারের সাথে সংযোগে সমস্যা হয়েছে। আবার চেষ্টা করুন।',

    // ── Voice ─────────────────────────────────────────────────────
    'voice.not_supported': 'ভয়েস রেকর্ডিং সমর্থিত নয়',
    'voice.start': 'ভয়েস রেকর্ডিং শুরু করতে ক্লিক করুন',
    'voice.stop': 'রেকর্ডিং বন্ধ করতে ক্লিক করুন',
    'voice.processing': 'ভয়েস প্রক্রিয়াকরণ হচ্ছে...',
    'voice.playing': 'প্লেব্যাক বন্ধ করতে ক্লিক করুন',
    'voice.error': 'ভয়েস ত্রুটি',
    'voice.mic_denied': 'মাইক্রোফোনের অনুমতি প্রত্যাখ্যাত। ব্রাউজার সেটিংসে অ্যাক্সেস অনুমতি দিন।',
    'voice.mic_not_found': 'কোনো মাইক্রোফোন পাওয়া যায়নি। একটি মাইক্রোফোন সংযুক্ত করুন।',

    // ── Sidebar ───────────────────────────────────────────────────
    'sidebar.patient_profile': 'রোগীর প্রোফাইল',
    'sidebar.age': 'বয়স',
    'sidebar.age_placeholder': 'যেমন ৩৫',
    'sidebar.gender': 'লিঙ্গ',
    'sidebar.gender_not_specified': 'উল্লেখ করা হয়নি',
    'sidebar.gender_male': 'পুরুষ',
    'sidebar.gender_female': 'মহিলা',
    'sidebar.gender_other': 'অন্যান্য',
    'sidebar.profile_hint': 'সঠিক পূর্বাভাসের জন্য প্রতিটি বার্তার সাথে প্রোফাইল পাঠানো হয়।',
    'sidebar.status': 'অবস্থা',
    'sidebar.peak_urgency': 'সর্বোচ্চ জরুরিতা:',
    'sidebar.stable': 'স্থিতিশীল',
    'sidebar.symptoms': 'রিপোর্ট করা লক্ষণ',
    'sidebar.clinical_state': 'ক্লিনিক্যাল অবস্থা',
    'sidebar.filled': 'পূরণ',
    'sidebar.no_clinical_data': 'এখনো কোনো ক্লিনিক্যাল তথ্য নেই।',
    'sidebar.predictions': 'শীর্ষ পূর্বাভাস',
    'sidebar.awaiting': 'তথ্যের অপেক্ষায়…',
    'sidebar.reports': 'রিপোর্ট টাইমলাইন',
    'sidebar.no_reports': 'কোনো রিপোর্ট আপলোড করা হয়নি।',
    'sidebar.findings_extracted': '⚠ ফলাফল নিষ্কাশিত',
    'sidebar.unknown_date': 'তারিখ অজানা',
    'sidebar.imaging': 'ইমেজিং স্টাডি',
    'sidebar.no_imaging': 'কোনো ইমেজিং স্টাডি আপলোড করা হয়নি।',
    'sidebar.no_abnormalities': '✓ কোনো অস্বাভাবিকতা নেই',
    'sidebar.confidence': 'নির্ভুলতা',

    // ── Settings ──────────────────────────────────────────────────
    'settings.title': 'সেটিংস',
    'settings.language': 'ভাষা',
    'settings.voice': 'ভয়েস',
    'settings.voice_male': 'পুরুষ',
    'settings.voice_female': 'মহিলা',
    'settings.speech_rate': 'কথার গতি',
    'settings.speed_slow': 'ধীর',
    'settings.speed_normal': 'স্বাভাবিক',
    'settings.speed_fast': 'দ্রুত',

    // ── Report Page ───────────────────────────────────────────────
    'report.title': 'মেডিকেল রিপোর্ট বিশ্লেষণ',
    'report.upload': 'রিপোর্ট আপলোড',
    'report.analyzing': 'রিপোর্ট বিশ্লেষণ করা হচ্ছে...',
    'report.error': 'রিপোর্ট বিশ্লেষণে সমস্যা হয়েছে। আবার চেষ্টা করুন।',
    'report.upload_xray': 'চেস্ট এক্স-রে আপলোড করুন (JPG/PNG) — MedGemma AI দ্বারা বিশ্লেষিত',
    'report.upload_report': 'মেডিকেল রিপোর্ট আপলোড করুন (PDF বা ছবি)',

    // ── Stage names ───────────────────────────────────────────────
    'stage.1': 'প্রধান অভিযোগ',
    'stage.2': 'বিবরণ',
    'stage.3': 'ঝুঁকির সংকেত',
    'stage.4': 'পার্থক্য নির্ণয়',
    'stage.5': 'পরবর্তী পদক্ষেপ',
  },
};

/**
 * Get a translated string for the given key and language.
 * Falls back to English, then to the raw key.
 */
export function translate(lang: SupportedLanguage, key: string): string {
  return translations[lang]?.[key] ?? translations.en[key] ?? key;
}

export default translations;
