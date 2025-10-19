import React from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, Heart, Droplet, Activity, Loader2 } from 'lucide-react';
import { AnalysisResult } from '../services/mlApi';

interface ResultPageProps {
  imageUrl: string;
  analysisType: string;
  onTryAnother: () => void;
  result: AnalysisResult | null;
  isAnalyzing?: boolean;
}

export function ResultPage({ imageUrl, analysisType, onTryAnother, result, isAnalyzing }: ResultPageProps) {
  // Use API results instead of mock data
  const isAnemia = analysisType === 'anemia';
  const detected = result?.detected ?? false;
  const confidence = result?.confidence ?? 0;
  const heatmapUrl = result?.heatmapUrl;

  const getResultIcon = () => {
    if (detected) {
      return <AlertTriangle className="w-12 h-12 text-amber-500" />;
    }
    return <CheckCircle className="w-12 h-12 text-green-500" />;
  };

  const getResultText = () => {
    if (isAnemia) {
      return detected ? 'Anemia Detected' : 'No Anemia Detected';
    }
    return detected ? 'Jaundice Detected' : 'Normal Result';
  };

  const getResultColor = () => {
    return detected ? 'amber' : 'green';
  };

  const healthTips = isAnemia
    ? [
        {
          icon: <Heart className="w-6 h-6 text-red-500" />,
          title: 'Iron-Rich Foods',
          description: 'Include spinach, red meat, beans, and fortified cereals in your diet',
        },
        {
          icon: <Droplet className="w-6 h-6 text-blue-500" />,
          title: 'Vitamin C',
          description: 'Consume citrus fruits to enhance iron absorption',
        },
        {
          icon: <Activity className="w-6 h-6 text-teal-500" />,
          title: 'Regular Checkups',
          description: 'Consult a healthcare provider for proper diagnosis and treatment',
        },
      ]
    : [
        {
          icon: <Heart className="w-6 h-6 text-red-500" />,
          title: 'Liver Health',
          description: 'Maintain liver health with a balanced diet and avoid excessive alcohol',
        },
        {
          icon: <Droplet className="w-6 h-6 text-blue-500" />,
          title: 'Hydration',
          description: 'Stay well-hydrated to support liver function',
        },
        {
          icon: <Activity className="w-6 h-6 text-teal-500" />,
          title: 'Medical Attention',
          description: 'Seek immediate medical care if jaundice is confirmed',
        },
      ];

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-6">
        <Button
          onClick={onTryAnother}
          variant="outline"
          className="border-teal-300 text-teal-700 hover:bg-teal-50"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Try Another Image
        </Button>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left Column: Image and Result */}
        <div className="space-y-6">
          {/* Uploaded Image */}
          <Card className="p-6 border-teal-100">
            <h3 className="mb-4 text-gray-700">Uploaded Image</h3>
            <img
              src={imageUrl}
              alt="Uploaded eye"
              className="w-full rounded-lg shadow-md"
            />
          </Card>

          {/* Grad-CAM Heatmap */}
          <Card className="p-6 border-teal-100">
            <h3 className="mb-4 text-gray-700">Grad-CAM Heatmap</h3>
            <div className="relative">
              <img
                src={heatmapUrl || imageUrl}
                alt="Heatmap"
                className="w-full rounded-lg shadow-md opacity-60"
              />
              <div className="absolute inset-0 bg-gradient-to-br from-red-500/40 via-yellow-500/40 to-transparent rounded-lg mix-blend-multiply" />
            </div>
            <p className="text-sm text-gray-600 mt-3">
              Heat map showing regions of interest analyzed by the AI model
            </p>
          </Card>
        </div>

        {/* Right Column: Results and Tips */}
        <div className="space-y-6">
          {/* Result Card */}
          <Card className={`p-8 border-2 ${detected ? 'border-amber-200 bg-amber-50/30' : 'border-green-200 bg-green-50/30'}`}>
            <div className="flex items-center gap-4 mb-6">
              {getResultIcon()}
              <div className="flex-1">
                <h2 className={`${detected ? 'text-amber-700' : 'text-green-700'} mb-2`}>
                  {getResultText()}
                </h2>
                <Badge
                  variant={detected ? 'destructive' : 'default'}
                  className={detected ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700'}
                >
                  {analysisType === 'anemia' ? 'Anemia Analysis' : 'Jaundice Analysis'}
                </Badge>
              </div>
            </div>

            {/* Confidence Score */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-700">Confidence Score</span>
                <span className={`${detected ? 'text-amber-700' : 'text-green-700'}`}>
                  {confidence.toFixed(1)}%
                </span>
              </div>
              <Progress value={confidence} className="h-3" />
            </div>

            <div className="mt-6 p-4 bg-white rounded-lg border border-gray-200">
              <p className="text-sm text-gray-600">
                {detected
                  ? 'The analysis suggests signs consistent with ' + (isAnemia ? 'anemia' : 'jaundice') + '. Please consult a healthcare professional for proper diagnosis.'
                  : 'The analysis shows normal results. Continue regular health monitoring.'}
              </p>
            </div>
          </Card>

          {/* Health Tips */}
          <Card className="p-6 border-teal-100">
            <h3 className="mb-4 text-teal-700">Health Tips & Recommendations</h3>
            <div className="space-y-4">
              {healthTips.map((tip, index) => (
                <div
                  key={index}
                  className="flex gap-4 p-4 bg-gradient-to-r from-teal-50 to-blue-50 rounded-lg"
                >
                  <div className="flex-shrink-0">{tip.icon}</div>
                  <div>
                    <h4 className="text-gray-800 mb-1">{tip.title}</h4>
                    <p className="text-sm text-gray-600">{tip.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}