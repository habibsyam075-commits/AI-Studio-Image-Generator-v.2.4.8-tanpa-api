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
  const sensualModeDirective = modelData.isSensual
    ? `\n\n  **Sensual Mode Directive:** The overall tone of the photograph should be intimate, tasteful, and sensual. CRITICAL: The lighting, pose, and composition must be expertly crafted to tastefully and artistically accentuate the subject's specified **${modelData.bodyShape}** body shape. This is the primary goal of the sensual mode. Avoid anything explicit or vulgar.`
    : '';

  const personaDirective = modelType === 'professional'
    ? `\n\n  **Subject Persona (CRITICAL):** The subject is a **Professional Model**. Their pose, expression, and gaze must reflect this. They are confident, skilled, and aware of the camera's presence. Their body language should be deliberate and composed.`
    : `\n\n  **Subject Persona (CRITICAL):** The subject is a **Normal Person**, not a model. The goal is to capture a genuine, candid moment. Their pose, expression, and body language must be completely natural, unposed, and relaxed. They should appear unaware of the camera or as if a friend is taking their picture.`;

  const ethnicFeaturesGuide = ETHNICITY_FEATURES_MAP[country] || 'A diverse range of human features.';
  const shotTypeDescription = SHOT_TYPE_DESCRIPTIONS[sceneData.shotType] || sceneData.shotType;
  const outfitColorDirective = modelData.outfitColor && modelData.outfitColor.toLowerCase() !== 'any' ? `CRITICAL: The dominant color of the outfit MUST be **${modelData.outfitColor}**.` : '';

  const prompt = `
  **//-- CORE DIRECTIVE: HYPER-REALISM --//**
  **Goal:** Generate a single, **photorealistic** image that is indistinguishable from a high-end digital photograph. The aesthetic must be raw, authentic, and detailed.
  **Seed:** ${Date.now()}

  **//-- NEGATIVE PROMPT (STRICTLY FORBIDDEN) --//**
  - **ABSOLUTELY NO:** Digital art, CGI, 3D rendering, video game graphics, illustration, painting, drawing, or anime styles.
  - **NO:** Plastic skin, airbrushed textures, wax-figure looks, perfect symmetry, or "stock photo" artificiality.
  - **NO:** Distorted limbs, extra fingers, or unnatural blurring.

  **//-- PHOTOGRAPHIC ENGINE SETTINGS --//**
  - **Camera:** Emulate a **Sony A7R V** or **Phase One XF IQ4**. 
  - **Lens:** **85mm f/1.2 GM** or **105mm f/1.4 Art**. Focus on creating a realistic shallow depth of field with creamy bokeh that separates the subject from the background.
  - **Skin Texture (HIGHEST PRIORITY):** You MUST render visible skin pores, vellus hair (peach fuzz), fine lines, and natural skin irregularities. Skin must NOT look smooth or plastic. Subsurface scattering is required for realistic lighting on skin.
  - **Imperfections:** Introduce subtle signs of reality—stray hairs, slight fabric creases, dust motes in light beams, or natural asymmetry. 

  ${sensualModeDirective}${personaDirective}

  **//-- CULTURAL CONTEXT: ${country} --//**
  - **Ethnicity:** The subject MUST authentically represent a person from **${country}**.
  - **Features:** Strictly follow: **"${ethnicFeaturesGuide}"**.
  - **Environment:** The location must feel like a real place in ${country}, not a generic set.

  **//-- SUBJECT DETAILS --//**
  - **Subject:** A ${modelData.age}-year-old ${modelData.gender}.
  - **Body:** ${modelData.description}. Shape: **${modelData.bodyShape}**.
  - **Expression:** ${modelData.expression}. Natural and unforced.
  - **Pose:** ${modelData.pose}.
  - **Outfit:** ${modelData.outfit}. ${outfitColorDirective}
  - **Style:** ${overallStyle} aesthetic.

  **//-- SCENE SETTINGS --//**
  - **Framing:** **${shotTypeDescription}**.
  - **Location:** ${sceneData.location}.
  - **Lighting:** ${sceneData.lighting}. Ensure physically accurate light behavior (shadows, bounce, occlusion).
  - **Mood:** ${sceneData.mood}.
  - **Details:** ${sceneData.details}.

  **//-- REFERENCE INSTRUCTIONS --//**
  ${referenceData.usePhoto && referenceData.photo ? `
    - Reference Image Provided.
    - Style Match: ${referenceData.useStyle ? 'High' : 'None'}.
    - Composition Match: ${referenceData.useComposition ? 'High' : 'None'}.
  ` : 'No reference image.'}
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
  const prompt = `You are a reality simulation engine. Your task is to adapt a preset scene concept into a 100% realistic data profile for an average, everyday location in ${country}.
  
  **Core Mandate: Absolute Realism, NOT Aesthetics (NON-NEGOTIABLE)**
  Your ONLY priority is raw, unpolished realism. AVOID glossy, idealized, or "influencer" aesthetics at all costs.

  **MANDATE FOR MUNDANITY (ABSOLUTE RULE):**
  - **AVOID:** Designer furniture, perfect cleanliness, trendy decor.
  - **INCLUDE:** Normal signs of wear and tear, generic non-designer items, unplanned clutter, and culturally specific *commonplace* items.

  **Adaptation Task:**
  - **Original Scene Concept:** ${presetData.location}
  - **Original Scene Details:** ${presetData.details}
  - **Target Country:** ${country}

  **Instruction (CRITICAL):**
  Take the *essence* of the preset and filter it through your realism engine for ${country}. If the preset is "Grandma's Kitchen", you MUST describe a REAL, slightly messy, lived-in kitchen of a typical, non-wealthy grandmother in ${country}. It must NOT be a stylized, perfectly clean "farmhouse chic" kitchen. It must have authentic, culturally specific clutter.

  Provide a JSON object with two keys: "location" and "details".
  - "location": The rewritten, realistic, culturally-adapted location description.
  - "details": The rewritten, realistic, culturally-adapted details with sensory information that ground the scene in unpolished reality.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            location: { type: Type.STRING, description: "The rewritten, culturally-adapted location description." },
            details: { type: Type.STRING, description: "The rewritten, culturally-adapted details with sensory information." },
          },
          required: ["location", "details"],
        },
      },
    });

    const jsonResponse = extractJson(response.text);
    return jsonResponse as Partial<SceneData>;
  } catch (error: any) {
    console.error("Error adapting scene preset:", error);
    throw new Error("Failed to adapt scene preset. The AI returned a response that could not be processed. Please try again.");
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

  const ethnicFeaturesGuide = ETHNICITY_FEATURES_MAP[country] || 'A diverse range of human features.';
  
  const sceneInstruction = sceneType === 'indoor'
    ? 'The scene MUST be an indoor location.'
    : 'The scene can be either indoor or outdoor.';

  const outfitStyle = currentModel.isSensual ? 'tasteful and sensual' : overallStyle;
  const outfitExamples = currentModel.isSensual ? SENSUAL_OUTFITS : (overallStyle === 'modern' ? MODERN_OUTFITS : AUTHENTIC_OUTFITS);

  const prompt = `
  You are a radical creative director for a high-volume, photorealistic image generation tool. Your ONLY task is to generate wildly creative, non-obvious, and contextually perfect values for the "unlocked" fields below. This is for a user generating thousands of images, so avoiding repetition is the absolute highest priority.

  **Current Photoshoot Context (Use this for inspiration and context):**
  - **Country for Ethnicity & Location:** ${country}
  - **Overall Style:** ${overallStyle}
  - **Model Persona:** ${modelType}
  - **Sensual Mode:** ${currentModel.isSensual}
  - **Current Model Details (For Avoidance):** ${JSON.stringify(currentModel)}
  - **Current Scene Details (For Avoidance):** ${JSON.stringify(currentScene)}

  **Your ONLY Task:**
  Generate new, creative values for the following fields ONLY: **${fieldsToRandomizeList}**.
  Your response MUST be a JSON object containing keys for ONLY these fields.

  **CRITICAL RULES for generating new values:**
  1.  **RADICAL DIVERGENCE:** The new values MUST be a massive creative leap from the current values. Subtle changes are a failure. Think "completely different photoshoot".
  2.  **AVOID CLICHÉS:** Actively reject the most common or stereotypical ideas associated with the context (${country}, ${overallStyle}). Find a unique, unexpected angle.
  3.  **CULTURAL & CONTEXTUAL AWARENESS:** All generated values must be authentic and appropriate for the given context (${country}, ${overallStyle}, ${modelType}, etc.).
  4.  **ETHNIC FEATURES (If 'description' or 'tones' are requested):** You MUST strictly adhere to this guide for authentic features: **"${ethnicFeaturesGuide}"**. Create a specific individual, do not repeat the guide.
  5.  **BODY SHAPE (If requested):** The body shape must be realistic and consistent with the overall description.
  6.  **OUTFIT (If requested):** The outfit MUST strictly match the '${outfitStyle}' style. It must also be culturally appropriate for ${country}. To ensure this, create a new, unique outfit description inspired by the following examples for a '${outfitStyle}' look, but DO NOT copy them directly: ${JSON.stringify(outfitExamples)}.
  7.  **OUTFIT COLOR (If requested):** Generate a creative and suitable color for the described outfit. It can be a simple color ('red') or more descriptive ('sky blue').
  8.  **POSE (If requested):** The pose must match the model persona (${modelType}) and sensuality (${currentModel.isSensual}).
  9.  **SCENE (If requested):** Scene elements must feel like a real, mundane, and culturally authentic place in ${country}. Avoid idealized or generic descriptions. ${sceneInstruction}

  Generate the JSON response now.
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
      console.error("Failed to parse JSON from smart randomization response. Original text:", response.text);
      throw new Error("Failed to process randomization. The AI returned a response that could not be read.");
  }
};

const pickRandom = <T>(options: T[], currentValue?: T): T => {
  const filteredOptions = currentValue ? options.filter(o => o !== currentValue) : options;
  if (filteredOptions.length > 0) {
    return filteredOptions[Math.floor(Math.random() * filteredOptions.length)];
  }
  // Fallback if all options are the same as current or options is empty
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