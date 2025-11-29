// services/geminiService.ts
import { GoogleGenAI, Part, Type } from '@google/genai';
import type { ModelData, SceneData, ReferenceData, ModelCreatorLocks, SceneLocks } from '../types';
import { GENDER_OPTIONS, EXPRESSION_OPTIONS, LIGHTING_OPTIONS, MOOD_OPTIONS, SHOT_TYPE_OPTIONS, ETHNICITY_FEATURES_MAP, SENSUAL_POSES, NON_SENSUAL_POSES, SHOT_TYPE_DESCRIPTIONS, MODERN_OUTFITS, AUTHENTIC_OUTFITS, SENSUAL_OUTFITS, RANDOM_DESCRIPTIONS, RANDOM_TONES, RANDOM_LOCATIONS, RANDOM_DETAILS, BODY_SHAPE_OPTIONS, RANDOM_COLORS } from '../constants';

const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.readAsDataURL(file);
  });
  return {
    inlineData: {
      data: await base64EncodedDataPromise,
      mimeType: file.type,
    },
  };
};

/**
 * Builds a detailed text prompt for image generation based on user inputs.
 */
const buildPrompt = (modelData: ModelData, sceneData: SceneData, country: string, referenceData: ReferenceData, overallStyle: 'modern' | 'authentic', modelType: 'professional' | 'natural'): string => {
  const personaDesc = modelType === 'professional' 
    ? 'Professional model, confident pose.' 
    : 'Candid, natural, unaware of camera.';
    
  const ethnicFeaturesGuide = ETHNICITY_FEATURES_MAP[country] || 'Diverse features.';
  const shotTypeDescription = SHOT_TYPE_DESCRIPTIONS[sceneData.shotType] || sceneData.shotType;
  const outfitColorDirective = modelData.outfitColor && modelData.outfitColor.toLowerCase() !== 'any' ? `, Color: ${modelData.outfitColor}` : '';

  // Compact, high-impact prompt structure for speed and realism
  const prompt = `
  [DIRECTIVE: PHOTOREALISM]
  Generate a RAW, AUTHENTIC, HIGH-RESOLUTION photograph.
  Camera: 85mm f/1.2. Focus on texture, visible skin pores, imperfections, natural lighting. 
  NO: CGI, illustration, 3D, plastic skin, airbrushed, smoothing, distortion, anime.

  [SUBJECT]
  ${modelData.age}yo ${modelData.gender}. ${modelData.description}. Body: ${modelData.bodyShape}.
  Expression: ${modelData.expression}. Pose: ${modelData.pose}.
  Outfit: ${modelData.outfit}${outfitColorDirective}.
  Ethnicity: ${country} (${ethnicFeaturesGuide}).
  Persona: ${personaDesc}
  ${modelData.isSensual ? 'Mood: Tasteful, sensual, intimate.' : ''}

  [SCENE]
  Location: ${sceneData.location}.
  Lighting: ${sceneData.lighting}.
  Mood: ${sceneData.mood}.
  Details: ${sceneData.details}.
  Framing: ${shotTypeDescription}.

  [STYLE]
  ${overallStyle} aesthetic.
  ${referenceData.usePhoto ? 'Use provided image as strict reference.' : ''}
  `;

  return prompt;
};

// Helper function for robust JSON extraction from AI responses
function extractJson(text: string): any {
  // Attempt to find JSON block within markdown ```json ... ```
  const markdownJsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
  if (markdownJsonMatch && markdownJsonMatch[1]) {
    try {
      return JSON.parse(markdownJsonMatch[1]);
    } catch (e) {
      console.error("Failed to parse JSON from markdown block, falling back.", e);
    }
  }

  // Fallback to finding the first '{' and last '}'
  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');

  if (firstBrace === -1 || lastBrace === -1 || lastBrace < firstBrace) {
    throw new Error("AI response did not contain a valid JSON object.");
  }

  const jsonString = text.substring(firstBrace, lastBrace + 1);
  try {
    return JSON.parse(jsonString);
  } catch (e) {
    console.error("Failed to parse extracted JSON string:", jsonString);
    throw new Error("AI returned malformed JSON.");
  }
}


/**
 * Generates an image using Gemini based on text and optional image inputs.
 * Returns an array of base64 encoded strings of the generated image(s).
 */
export const generateAIImage = async (
  modelData: ModelData,
  sceneData: SceneData,
  referenceData: ReferenceData,
  country: string,
  overallStyle: 'modern' | 'authentic',
  modelType: 'professional' | 'natural',
  aspectRatio: '1:1' | '3:4' | '9:16',
  numberOfImages: 1 | 4 = 1,
  generationTier: 'premium' | 'standard' = 'premium',
  imageSize: '1K' | '2K' | '4K' = '1K',
): Promise<string[]> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const prompt = buildPrompt(modelData, sceneData, country, referenceData, overallStyle, modelType);

  // Case 1: Reference photo is provided (Image Editing/Style Transfer)
  // Use gemini-2.5-flash-image, which generates a single image.
  if (referenceData.usePhoto && referenceData.photo) {
    const parts: Part[] = [{ text: prompt }];
    const imagePart = await fileToGenerativePart(referenceData.photo);
    parts.unshift(imagePart);

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts },
      });

      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            return [part.inlineData.data]; // Return as an array with one image
          }
        }
      } else {
        const finishReason = response.candidates?.[0]?.finishReason;
        if (finishReason) {
            throw new Error(`Image generation failed. Reason: ${finishReason}. Please adjust your prompt.`);
        }
      }
    } catch (error: any) {
      console.error("Error generating image with Gemini:", error);
      throw new Error(error.message || 'An unknown error occurred during image generation.');
    }

    throw new Error('No image was generated by the AI. Please try adjusting your prompt or reference image.');
  } 
  
  // Case 2: No reference photo (Image Generation)
  else {
    if (generationTier === 'standard') {
      // Use gemini-2.5-flash-image for "tier 1" generation
      try {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: {
            parts: [{ text: prompt }],
          },
          config: {
              imageConfig: {
                  aspectRatio: aspectRatio,
              },
          },
        });

        if (response.candidates?.[0]?.content?.parts) {
          for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
              return [part.inlineData.data]; // Return as an array with one image
            }
          }
        } else {
          const finishReason = response.candidates?.[0]?.finishReason;
          if (finishReason) {
            throw new Error(`Image generation failed. Reason: ${finishReason}. This may be due to safety policies.`);
          }
        }
        
        throw new Error('Image generation failed with Standard engine. No image data received.');

      } catch (error: any) {
        console.error("Error generating image with Gemini Flash Image:", error);
        throw new Error(error.message || 'An unknown error occurred during image generation with the Standard engine.');
      }
    } else { // 'premium' tier
      // Use gemini-3-pro-image-preview to generate 1 or more images.
      try {
        const generateSingleImage = async (): Promise<string> => {
            const response = await ai.models.generateContent({
                model: 'gemini-3-pro-image-preview',
                contents: { parts: [{ text: prompt }] },
                config: {
                    imageConfig: {
                        aspectRatio: aspectRatio,
                        imageSize: imageSize,
                    },
                },
            });

            if (response.candidates?.[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if (part.inlineData) {
                        return part.inlineData.data;
                    }
                }
            }
            const finishReason = response.candidates?.[0]?.finishReason;
            if (finishReason) {
              throw new Error(`Image generation failed. Reason: ${finishReason}. This may be due to safety policies.`);
            }
            throw new Error('Image generation failed. The model did not return any images.');
        };

        if (numberOfImages === 1) {
            const imageData = await generateSingleImage();
            return [imageData];
        } else {
            const imagePromises = Array.from({ length: numberOfImages }, () => generateSingleImage());
            const imagesData = await Promise.all(imagePromises);
            return imagesData;
        }
      } catch (error: any)
      {
        console.error("Error generating images with Gemini Pro Image:", error);
        throw new Error(error.message || 'An unknown error occurred during image generation with the Premium engine.');
      }
    }
  }
};

/**
 * Adapts a scene preset to a specific country's cultural context using Gemini.
 */
export const adaptScenePreset = async (
  presetData: Partial<SceneData>,
  country: string
): Promise<Partial<SceneData>> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const prompt = `Adapt this scene for ${country}. REALISTIC, MUNDANE, LIVED-IN.
  Original: ${presetData.location}. ${presetData.details}
  Return JSON { "location": "...", "details": "..." } with culturally authentic, non-idealized descriptions.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            location: { type: Type.STRING },
            details: { type: Type.STRING },
          },
          required: ["location", "details"],
        },
      },
    });

    const jsonResponse = extractJson(response.text);
    return jsonResponse as Partial<SceneData>;
  } catch (error: any) {
    console.error("Error adapting scene preset:", error);
    throw new Error("Failed to adapt scene preset.");
  }
};

/**
 * Intelligently randomizes unlocked fields in a single, efficient API call.
 */
export const generateSmartRandomization = async (
  unlockedFields: { model: (keyof Omit<ModelCreatorLocks, 'all'>)[]; scene: (keyof Omit<SceneLocks, 'all'>)[]; },
  currentModel: ModelData,
  currentScene: SceneData,
  country: string,
  overallStyle: 'modern' | 'authentic',
  modelType: 'professional' | 'natural',
  sceneType: 'any' | 'indoor' = 'any'
): Promise<Partial<ModelData & SceneData>> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const { model: unlockedModel, scene: unlockedScene } = unlockedFields;

  const fieldsToRandomizeList = [
    ...unlockedModel.map(f => `model.${f}`),
    ...unlockedScene.map(f => `scene.${f}`)
  ].join(', ');
  
  if (!fieldsToRandomizeList) {
    return {};
  }

  const properties: Record<string, { type: any; enum?: readonly string[]; }> = {};
  if (unlockedModel.includes('description')) properties['description'] = { type: Type.STRING };
  if (unlockedModel.includes('gender')) properties['gender'] = { type: Type.STRING, enum: GENDER_OPTIONS };
  if (unlockedModel.includes('age')) properties['age'] = { type: Type.INTEGER };
  if (unlockedModel.includes('expression')) properties['expression'] = { type: Type.STRING, enum: EXPRESSION_OPTIONS };
  if (unlockedModel.includes('bodyShape')) properties['bodyShape'] = { type: Type.STRING, enum: BODY_SHAPE_OPTIONS };
  if (unlockedModel.includes('outfit')) properties['outfit'] = { type: Type.STRING };
  if (unlockedModel.includes('outfitColor')) properties['outfitColor'] = { type: Type.STRING };
  if (unlockedModel.includes('tones')) properties['tones'] = { type: Type.STRING };
  if (unlockedModel.includes('pose')) properties['pose'] = { type: Type.STRING };
  if (unlockedScene.includes('location')) properties['location'] = { type: Type.STRING };
  if (unlockedScene.includes('lighting')) properties['lighting'] = { type: Type.STRING, enum: LIGHTING_OPTIONS };
  if (unlockedScene.includes('mood')) properties['mood'] = { type: Type.STRING, enum: MOOD_OPTIONS };
  if (unlockedScene.includes('details')) properties['details'] = { type: Type.STRING };

  const responseSchema = {
    type: Type.OBJECT,
    properties,
  };

  const ethnicFeaturesGuide = ETHNICITY_FEATURES_MAP[country] || 'Diverse features.';
  
  const prompt = `
  Context: ${country}, ${overallStyle}, ${modelType}.
  Task: Creative randomization for fields: ${fieldsToRandomizeList}.
  Rules:
  1. Unique, non-cliché values.
  2. Culturally authentic to ${country}.
  3. ${unlockedModel.includes('outfit') ? `Outfit: ${overallStyle} style.` : ''}
  4. Features: ${ethnicFeaturesGuide}.
  Generate JSON.
  `;
  
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
    config: {
      responseMimeType: 'application/json',
      responseSchema: responseSchema,
    },
  });

  try {
    const jsonResponse = extractJson(response.text);
    return jsonResponse as Partial<ModelData & SceneData>;
  } catch (error) {
      console.error("Failed to parse JSON from smart randomization response.", error);
      throw new Error("Failed to process randomization.");
  }
};

const pickRandom = <T>(options: T[], currentValue?: T): T => {
  const filteredOptions = currentValue ? options.filter(o => o !== currentValue) : options;
  if (filteredOptions.length > 0) {
    return filteredOptions[Math.floor(Math.random() * filteredOptions.length)];
  }
  return options[Math.floor(Math.random() * options.length)];
};

export const generateLocalRandomization = (
  unlockedFields: { model: (keyof Omit<ModelCreatorLocks, 'all'>)[]; scene: (keyof Omit<SceneLocks, 'all'>)[]; },
  currentModel: ModelData,
  currentScene: SceneData,
  overallStyle: 'modern' | 'authentic'
): Partial<ModelData & SceneData> => {
  const result: Partial<ModelData & SceneData> = {};
  const { model: unlockedModel, scene: unlockedScene } = unlockedFields;

  // Model randomization
  if (unlockedModel.includes('description')) result.description = pickRandom(RANDOM_DESCRIPTIONS, currentModel.description);
  if (unlockedModel.includes('gender')) result.gender = pickRandom(GENDER_OPTIONS, currentModel.gender);
  if (unlockedModel.includes('age')) result.age = Math.floor(Math.random() * (60 - 18 + 1)) + 18; // 18-60
  if (unlockedModel.includes('expression')) result.expression = pickRandom(EXPRESSION_OPTIONS, currentModel.expression);
  if (unlockedModel.includes('bodyShape')) result.bodyShape = pickRandom(BODY_SHAPE_OPTIONS, currentModel.bodyShape);
  if (unlockedModel.includes('outfit')) {
    const outfitOptions = currentModel.isSensual
      ? SENSUAL_OUTFITS
      : (overallStyle === 'modern' ? MODERN_OUTFITS : AUTHENTIC_OUTFITS);
    result.outfit = pickRandom(outfitOptions, currentModel.outfit);
  }
  if (unlockedModel.includes('outfitColor')) result.outfitColor = pickRandom(RANDOM_COLORS, currentModel.outfitColor);
  if (unlockedModel.includes('tones')) result.tones = pickRandom(RANDOM_TONES, currentModel.tones);
  if (unlockedModel.includes('pose')) {
    const poseOptions = currentModel.isSensual ? SENSUAL_POSES : NON_SENSUAL_POSES;
    result.pose = pickRandom(poseOptions, currentModel.pose);
  }

  // Scene randomization
  if (unlockedScene.includes('location')) result.location = pickRandom(RANDOM_LOCATIONS, currentScene.location);
  if (unlockedScene.includes('lighting')) result.lighting = pickRandom(LIGHTING_OPTIONS, currentScene.lighting);
  if (unlockedScene.includes('mood')) result.mood = pickRandom(MOOD_OPTIONS, currentScene.mood);
  if (unlockedScene.includes('details')) result.details = pickRandom(RANDOM_DETAILS, currentScene.details);

  return result;
};
