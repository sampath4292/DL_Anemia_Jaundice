import React, { useState } from 'react';
import { HomePage } from './components/HomePage';
import { ResultPage } from './components/ResultPage';
import { SessionHistorySidebar } from './components/SessionHistorySidebar';
import { LanguageSwitcher } from './components/LanguageSwitcher';
import { Footer } from './components/Footer';
import { Menu, X } from 'lucide-react';
import { Button } from './components/ui/button';
import { performAnalysis, AnalysisResult } from './services/mlApi';
import { toast } from 'sonner@2.0.3';

interface HistoryItem {
  id: string;
  imageUrl: string;
  analysisType: string;
  result: string;
  timestamp: string;
  detected: boolean;
  confidence: number;
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'result'>('home');
  const [language, setLanguage] = useState('en');
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [currentAnalysisType, setCurrentAnalysisType] = useState<string>('anemia');
  const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = async (file: File, analysisType: string) => {
    try {
      setIsAnalyzing(true);
      const imageUrl = URL.createObjectURL(file);
      setCurrentImage(imageUrl);
      setCurrentAnalysisType(analysisType);

      // Call the ML API
      const result = await performAnalysis(file, analysisType as 'anemia' | 'jaundice');
      setCurrentResult(result);
      setCurrentPage('result');

      // Add to history
      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        imageUrl,
        analysisType: analysisType === 'anemia' ? 'Anemia' : 'Jaundice',
        result: result.detected
          ? `${analysisType === 'anemia' ? 'Anemia' : 'Jaundice'} Detected`
          : 'Normal',
        timestamp: new Date().toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
        detected: result.detected,
        confidence: result.confidence,
      };

      setHistory((prev) => [newHistoryItem, ...prev]);
      toast.success('Analysis completed successfully');
    } catch (error) {
      console.error('Analysis failed:', error);
      toast.error('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTryAnother = () => {
    setCurrentPage('home');
    setCurrentImage(null);
    setCurrentResult(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-50 via-teal-50/30 to-blue-50/30">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="icon"
                className="lg:hidden"
                onClick={() => setSidebarOpen(!sidebarOpen)}
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 bg-gradient-to-br from-teal-500 to-blue-500 rounded-lg flex items-center justify-center">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    className="w-6 h-6 text-white"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-teal-700">EyeHealth AI</h1>
                </div>
              </div>
            </div>
            <LanguageSwitcher language={language} onLanguageChange={setLanguage} />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Sidebar for desktop */}
        <aside className="hidden lg:block w-80 bg-white border-r border-gray-200 overflow-y-auto">
          <SessionHistorySidebar history={history} />
        </aside>

        {/* Mobile Sidebar */}
        {sidebarOpen && (
          <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setSidebarOpen(false)}>
            <div
              className="w-80 h-full bg-white overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-4 border-b border-gray-200 flex items-center justify-between">
                <h3 className="text-gray-700">Session History</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setSidebarOpen(false)}
                >
                  <X className="w-5 h-5" />
                </Button>
              </div>
              <SessionHistorySidebar history={history} />
            </div>
          </div>
        )}

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto">
          {currentPage === 'home' ? (
            <HomePage onAnalyze={handleAnalyze} />
          ) : (
            currentImage && (
              <ResultPage
                imageUrl={currentImage}
                analysisType={currentAnalysisType}
                onTryAnother={handleTryAnother}
                result={currentResult}
                isAnalyzing={isAnalyzing}
              />
            )
          )}
        </main>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
}