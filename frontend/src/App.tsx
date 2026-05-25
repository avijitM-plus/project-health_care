import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChatProvider } from './context/ChatContext';
import { Sidebar } from './components/Sidebar';
import { ChatPage } from './pages/ChatPage';
import { ReportPage } from './pages/ReportPage';

const App: React.FC = () => {
  return (
    <Router>
      <ChatProvider>
        <div className="app-container">
          <Sidebar />
          <div className="main-content">
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/report" element={<ReportPage />} />
            </Routes>
          </div>
        </div>
      </ChatProvider>
    </Router>
  );
};

export default App;
