// ML Model API Service
// Replace the API_URL with your actual ML model endpoint

const API_URL = "http://localhost:8000"; // local FastAPI backend

export interface AnalysisResult {
  detected: boolean;
  confidence: number;
  analysisType: "anemia" | "jaundice";
  heatmapUrl?: string; // Optional: URL to Grad-CAM heatmap image
  metadata?: {
    modelVersion?: string;
    processingTime?: number;
    [key: string]: any;
  };
}

/**
 * Analyzes an eye image for anemia or jaundice detection
 * @param imageFile - The eye image file to analyze
 * @param analysisType - Type of analysis: 'anemia' or 'jaundice'
 * @returns Promise with analysis results
 */
export async function analyzeImage(
  imageFile: File,
  analysisType: "anemia" | "jaundice"
): Promise<AnalysisResult> {
  try {
    // Create FormData to send the image
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("analysis_type", analysisType);

    // Make API request to /predict (no heatmap)
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
      headers: {
        // Add any required headers (e.g., API key)
        // 'Authorization': `Bearer ${YOUR_API_KEY}`,
      },
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();

    // If we expect a heatmap, request it separately from /gradcam
    let heatmapUrl: string | undefined = undefined;
    if (analysisType === "jaundice") {
      try {
        const gres = await fetch(`${API_URL}/gradcam`, {
          method: "POST",
          body: formData,
        });
        if (gres.ok) {
          const gdata = await gres.json();
          if (gdata.png_bytes_base64) {
            heatmapUrl = `data:image/png;base64,${gdata.png_bytes_base64}`;
          } else if (gdata.heatmap_url) {
            heatmapUrl = gdata.heatmap_url;
          }
        }
      } catch (e) {
        console.warn("Failed to fetch gradcam:", e);
      }
    }

    return {
      detected: !!data.detected,
      confidence: data.confidence || 0,
      analysisType: analysisType,
      heatmapUrl,
      metadata: {
        modelVersion: data.model_version,
        processingTime: data.processing_time,
        ...data.metadata,
      },
    };
  } catch (error) {
    console.error("Error analyzing image:", error);
    throw error;
  }
}

/**
 * Mock function for testing without a real API
 * Remove this when you connect to the real API
 */
export async function mockAnalyzeImage(
  imageFile: File,
  analysisType: "anemia" | "jaundice"
): Promise<AnalysisResult> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Mock response
  return {
    detected: Math.random() > 0.5,
    confidence: 85 + Math.random() * 10,
    analysisType: analysisType,
    heatmapUrl: undefined, // Will use the uploaded image with overlay
    metadata: {
      modelVersion: "1.0.0",
      processingTime: 1.5,
    },
  };
}

// Export the function you want to use
// Switch to analyzeImage when you have a real API
export const performAnalysis = analyzeImage; // Use real API
