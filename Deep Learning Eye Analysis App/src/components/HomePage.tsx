import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Upload, Eye } from 'lucide-react';

interface HomePageProps {
  onAnalyze: (file: File, analysisType: string) => void;
}

export function HomePage({ onAnalyze }: HomePageProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysisType, setAnalysisType] = useState<string>('anemia');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      onAnalyze(selectedFile, analysisType);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-12">
        <h1 className="text-teal-600 mb-3">
          Deep Learning-Based Eye Image Analysis
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Advanced AI-powered detection of Anemia and Jaundice through eye image analysis
        </p>
      </div>

      <div className="grid lg:grid-cols-[1fr,300px] gap-8 items-start">
        <div className="space-y-6">
          <Card className="p-8 border-2 border-teal-100">
            <div className="space-y-6">
              {/* Upload Section */}
              <div>
                <label className="block mb-3 text-gray-700">
                  Upload Eye Image
                </label>
                <div className="border-2 border-dashed border-teal-300 rounded-xl p-8 text-center hover:border-teal-400 transition-colors bg-teal-50/30">
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    {previewUrl ? (
                      <div className="space-y-3">
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="max-h-48 mx-auto rounded-lg object-cover"
                        />
                        <p className="text-sm text-teal-600">Click to change image</p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <Upload className="w-12 h-12 mx-auto text-teal-500" />
                        <div>
                          <p className="text-teal-700">Click to upload or drag and drop</p>
                          <p className="text-sm text-gray-500 mt-1">PNG, JPG up to 10MB</p>
                        </div>
                      </div>
                    )}
                  </label>
                </div>
              </div>

              {/* Analysis Type Selection */}
              <div>
                <label className="block mb-3 text-gray-700">
                  Select Analysis Type
                </label>
                <Select value={analysisType} onValueChange={setAnalysisType}>
                  <SelectTrigger className="w-full border-teal-200">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="anemia">Anemia Detection</SelectItem>
                    <SelectItem value="jaundice">Jaundice Detection</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Analyze Button */}
              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile}
                className="w-full bg-teal-600 hover:bg-teal-700 text-white py-6 rounded-xl disabled:opacity-50"
              >
                <Eye className="w-5 h-5 mr-2" />
                Analyze Image
              </Button>
            </div>
          </Card>

          {/* Information Cards */}
          <div className="grid sm:grid-cols-2 gap-4">
            <Card className="p-6 bg-gradient-to-br from-teal-50 to-blue-50 border-teal-100">
              <h3 className="text-teal-700 mb-2">Anemia Detection</h3>
              <p className="text-sm text-gray-600">
                Analyzes conjunctiva pallor and sclera coloration to detect signs of anemia
              </p>
            </Card>
            <Card className="p-6 bg-gradient-to-br from-blue-50 to-teal-50 border-blue-100">
              <h3 className="text-blue-700 mb-2">Jaundice Detection</h3>
              <p className="text-sm text-gray-600">
                Identifies yellowing of the sclera indicative of elevated bilirubin levels
              </p>
            </Card>
          </div>
        </div>

        {/* Eye Illustration */}
        <Card className="p-6 bg-gradient-to-br from-teal-50 to-white border-teal-100 hidden lg:block">
          <img
            src="https://images.unsplash.com/photo-1671810456796-48fd984dd413?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxodW1hbiUyMGV5ZSUyMGNsb3NlJTIwdXAlMjBtZWRpY2FsfGVufDF8fHx8MTc2MDg3MTUyOHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
            alt="Eye illustration"
            className="w-full h-auto rounded-lg shadow-lg"
          />
          <div className="mt-4 text-center">
            <p className="text-sm text-gray-600">
              Advanced AI technology for non-invasive health screening
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
