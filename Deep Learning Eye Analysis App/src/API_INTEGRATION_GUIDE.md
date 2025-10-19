# ML API Integration Guide

This guide explains how to connect your ML model API to the EyeHealth AI web application.

## Files to Modify

### 1. `/services/mlApi.ts` - Main API Configuration

This is the **primary file** where you'll connect your ML model API.

#### Steps:

1. **Update the API endpoint:**
   ```typescript
   const API_URL = 'YOUR_API_ENDPOINT_HERE'; 
   // Example: 'https://your-api.com/api/v1'
   // or 'https://your-domain.com:8000'
   ```

2. **Add authentication (if needed):**
   ```typescript
   headers: {
     'Authorization': `Bearer ${YOUR_API_KEY}`,
     // or 'X-API-Key': 'your-api-key-here'
   }
   ```

3. **Adjust the request format** to match your API:
   ```typescript
   // Current format sends:
   formData.append('image', imageFile);
   formData.append('analysis_type', analysisType);
   
   // Modify field names if your API expects different names
   // Example:
   formData.append('file', imageFile);
   formData.append('type', analysisType);
   ```

4. **Map the API response** to match the expected format:
   ```typescript
   return {
     detected: data.detected || data.prediction === 'positive',
     confidence: data.confidence || data.confidence_score || 0,
     analysisType: analysisType,
     heatmapUrl: data.heatmap_url || data.gradcam_url,
     metadata: {
       modelVersion: data.model_version,
       processingTime: data.processing_time,
       ...data.metadata,
     },
   };
   ```

5. **Switch from mock to real API:**
   ```typescript
   // Change this line at the bottom of mlApi.ts:
   export const performAnalysis = analyzeImage; // Instead of mockAnalyzeImage
   ```

## Expected API Response Format

Your API should return JSON in this format (adjust the mapping in step 4 if different):

```json
{
  "detected": true,
  "confidence": 87.5,
  "heatmap_url": "https://your-api.com/heatmaps/abc123.png",
  "model_version": "1.0.0",
  "processing_time": 1.5,
  "metadata": {
    // any additional data
  }
}
```

## Common API Configurations

### Python Flask/FastAPI Backend

```typescript
const API_URL = 'http://localhost:5000'; // or your deployed URL

// In analyzeImage function:
const response = await fetch(`${API_URL}/analyze`, {
  method: 'POST',
  body: formData,
  // Don't set Content-Type header - browser sets it automatically for FormData
});
```

### TensorFlow Serving

```typescript
const API_URL = 'http://localhost:8501/v1/models/eye_model:predict';

// You may need to convert image to base64 or tensor format
// Adjust the request body accordingly
```

### AWS SageMaker

```typescript
const API_URL = 'https://runtime.sagemaker.region.amazonaws.com/endpoints/your-endpoint';

headers: {
  'Content-Type': 'application/json',
  'X-Amzn-SageMaker-Custom-Attributes': 'accept_eula=true',
}
```

### Google Cloud AI Platform

```typescript
const API_URL = 'https://ml.googleapis.com/v1/projects/PROJECT_ID/models/MODEL_NAME:predict';

headers: {
  'Authorization': `Bearer ${GOOGLE_CLOUD_API_KEY}`,
  'Content-Type': 'application/json',
}
```

## Testing

1. **Use mock mode first** (default):
   - The app is currently using `mockAnalyzeImage` which simulates API responses
   - Test the UI flow before connecting to real API

2. **Test your API separately**:
   ```bash
   # Example curl test
   curl -X POST http://localhost:5000/analyze \
     -F "image=@test_eye.jpg" \
     -F "analysis_type=anemia"
   ```

3. **Enable real API**:
   - Update `performAnalysis` export in `/services/mlApi.ts`
   - Monitor browser console for errors
   - Check network tab in DevTools

## Error Handling

The app automatically handles errors with toast notifications. Common issues:

- **CORS errors**: Enable CORS on your API server
- **Network errors**: Check API URL and connectivity
- **401/403 errors**: Verify API key/authentication
- **500 errors**: Check API logs for server-side issues

## Environment Variables (Optional)

For production, consider using environment variables:

```typescript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
const API_KEY = import.meta.env.VITE_API_KEY;
```

## Files Updated

The following files have been updated to support API integration:

1. ✅ `/services/mlApi.ts` - API service (main file to edit)
2. ✅ `/App.tsx` - Uses API service, handles loading states
3. ✅ `/components/ResultPage.tsx` - Displays API results
4. ✅ `/components/HomePage.tsx` - Already configured (no changes needed)

## Next Steps

1. Configure your API endpoint in `/services/mlApi.ts`
2. Test with mock data first
3. Switch to real API by changing the export
4. Deploy your API backend
5. Update the API_URL to your production endpoint
