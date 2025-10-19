import React from 'react';
import { Card } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { CheckCircle, AlertTriangle, Clock } from 'lucide-react';

interface HistoryItem {
  id: string;
  imageUrl: string;
  analysisType: string;
  result: string;
  timestamp: string;
  detected: boolean;
}

interface SessionHistorySidebarProps {
  history: HistoryItem[];
  onSelectHistory?: (item: HistoryItem) => void;
}

export function SessionHistorySidebar({ history, onSelectHistory }: SessionHistorySidebarProps) {
  if (history.length === 0) {
    return (
      <div className="p-6 text-center text-gray-500">
        <Clock className="w-8 h-8 mx-auto mb-2 text-gray-400" />
        <p className="text-sm">No analysis history yet</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-4 space-y-3">
        <h3 className="text-gray-700 mb-4">Session History</h3>
        {history.map((item) => (
          <Card
            key={item.id}
            className="p-3 border-teal-100 hover:border-teal-300 cursor-pointer transition-all hover:shadow-md"
            onClick={() => onSelectHistory?.(item)}
          >
            <div className="flex gap-3">
              <img
                src={item.imageUrl}
                alt="Analysis thumbnail"
                className="w-16 h-16 rounded-md object-cover flex-shrink-0"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {item.detected ? (
                    <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                  )}
                  <span className="text-sm text-gray-800 truncate">
                    {item.result}
                  </span>
                </div>
                <Badge
                  variant="outline"
                  className="text-xs mb-1 bg-teal-50 text-teal-700 border-teal-200"
                >
                  {item.analysisType}
                </Badge>
                <p className="text-xs text-gray-500">{item.timestamp}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </ScrollArea>
  );
}
