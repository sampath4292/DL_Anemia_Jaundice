import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Globe } from 'lucide-react';

interface LanguageSwitcherProps {
  language: string;
  onLanguageChange: (lang: string) => void;
}

export function LanguageSwitcher({ language, onLanguageChange }: LanguageSwitcherProps) {
  return (
    <div className="flex items-center gap-2">
      <Globe className="w-4 h-4 text-teal-600" />
      <Select value={language} onValueChange={onLanguageChange}>
        <SelectTrigger className="w-[100px] border-teal-200">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="en">English</SelectItem>
          <SelectItem value="hi">हिंदी</SelectItem>
          <SelectItem value="te">తెలుగు</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
