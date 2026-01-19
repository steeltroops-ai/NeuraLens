# MediLens Video Demo Script

> **Total Duration:** ~10 minutes  
> **Format:** Screen recording with voiceover  
> **Resolution:** 1920x1080 @ 60fps

---

## Scene Overview

| Scene | Duration | Content | Screen |
|-------|----------|---------|--------|
| 1 | 0:00-0:45 | Opening Hook & Problem | Title Card + Stats |
| 2 | 0:45-1:30 | Vision & Solution | Homepage |
| 3 | 1:30-2:15 | Platform Introduction | Dashboard |
| 4 | 2:15-3:30 | Retinal Pipeline Demo | RetinaScan Page |
| 5 | 3:30-4:45 | Speech Pipeline Demo | SpeechMD Page |
| 6 | 4:45-5:45 | Cardiology Pipeline Demo | CardioPredict Page |
| 7 | 5:45-6:30 | AI Explanation + Voice | Explanation Panel |
| 8 | 6:30-7:15 | Architecture Overview | GitHub ARCHITECTURE.md |
| 9 | 7:15-7:45 | AI Chatbot Demo | Medical Chatbot |
| 10 | 7:45-8:30 | Additional Features | Dashboard Tour |
| 11 | 8:30-9:15 | Coming Soon & Roadmap | Sidebar + Vision |
| 12 | 9:15-10:00 | Closing & Impact | Homepage + Logo |

---

## SCENE 1: Opening Hook (0:00 - 0:45)

**SCREEN:** Black background with animated text appearing

**[0:00-0:15]** *Slow fade in*

> Right now, somewhere in a rural clinic, a patient is waiting.
>
> Waiting for a specialist who visits once a month.
> Waiting for a scan result that sits in a backlog for weeks.

**[0:15-0:30]** *Statistics appear on screen*

> Healthcare access is not just a policy problem.
> It is a data bottleneck.
>
> The expertise to analyze a retinal scan, interpret an ECG, or detect neurological signs in a voice... is scarce. Expensive. Geographically concentrated.

**[0:30-0:45]** *MediLens logo fade in*

> What if we could decouple diagnostic precision from the limitations of time and geography?
>
> Welcome to **MediLens**.

---

## SCENE 2: Vision & Solution (0:45 - 1:30)

**SCREEN:** Homepage - localhost:3000

**ACTION:** Open homepage, hover over hero section

**[0:45-1:00]**

> MediLens is not just another chatbot or a simple API wrapper.
>
> This is a production-grade, multimodal medical AI platform designed to ingest, analyze, and synthesize complex physiological data across the human body.

**[1:00-1:15]** *Scroll to features section*

> We built a system that sees the patient as a whole.
>
> A cardiologist doesn't just look at an ECG; they listen to breathing.
> A neurologist doesn't just look at an MRI; they listen to speech patterns.
>
> Why should our AI be any different?

**[1:15-1:30]** *Scroll to show technology logos*

> We integrate retinal imaging, acoustic analysis, cardiac signal processing, and chest X-ray analysis... all orchestrated into a single, unified workflow.

---

## SCENE 3: Dashboard Introduction (1:30 - 2:15)

**SCREEN:** Dashboard - localhost:3000/dashboard

**ACTION:** Log in with Clerk, show dashboard

**[1:30-1:45]** *Dashboard loads with greeting*

> Here is the MediLens Dashboard.
>
> Notice the personalized greeting. "Good afternoon, [User]."
> We designed this for clinicians who need instant context.

**[1:45-2:00]** *Hover over diagnostic cards*

> You can see all nine available AI diagnostic modules.
>
> RetinaScan AI, SpeechMD AI, CardioPredict AI, ChestXplorer AI...
> Plus SkinSense, Motor Assessment, Cognitive Testing, Multi-Modal, and NRI Fusion.

**[2:00-2:15]** *Show system status and user overview*

> On the right, we have your personal health overview.
> Total assessments, health score, last test date, and current risk level.
>
> Below that, recent activity shows your assessment history with risk indicators.

---

## SCENE 4: Retinal Pipeline Demo (2:15 - 3:30)

**SCREEN:** RetinaScan AI - localhost:3000/dashboard/retinal

**ACTION:** Click RetinaScan, upload fundus image, run analysis

**[2:15-2:30]** *Navigate to RetinaScan page*

> Let's dive into our flagship imaging pipeline: **RetinaScan AI**.
>
> Vascular health is systemic. Changes in the eye often reflect changes in the heart and brain.

**[2:30-2:45]** *Upload a fundus image*

> When a clinician uploads a fundus image, the system kicks off a multi-step process.
>
> First, the image hits our **Quality Gate**. We detect blur, uneven illumination, and artifacts.
> If quality drops below threshold, we reject it instantly.

**[2:45-3:00]** *Show analysis running with pipeline status bar*

> Watch the status bar at the bottom.
> You can see each stage: Uploading, Preprocessing, Segmenting Vessels, Extracting Biomarkers, Generating Heatmap.
>
> This is real-time pipeline observability.

**[3:00-3:15]** *Results appear with heatmap*

> Look at the result.
> We don't just output "Disease Detected."
>
> We generate a biological heatmap, a Grad-CAM visualization showing exactly which micro-aneurysms, exudates, or hemorrhages triggered the risk score.

**[3:15-3:30]** *Hover over biomarker cards*

> We extract 12 distinct biomarkers: Vessel Density, Cup-to-Disc Ratio, Hemorrhage Count, Exudates, and more.
>
> Each with a value, a status, and a clinical explanation.

---

## SCENE 5: Speech Pipeline Demo (3:30 - 4:45)

**SCREEN:** SpeechMD AI - localhost:3000/dashboard/speech

**ACTION:** Navigate to Speech page, record or upload audio, run analysis

**[3:30-3:45]** *Navigate to Speech Analysis*

> Now something arguably more innovative: our **Speech Analysis Pipeline**.
>
> The human voice is a rich, underutilized biomarker.
> Early stages of Parkinson's or Alzheimer's often manifest in vocal patterns years before physical tremors appear.

**[3:45-4:00]** *Click record or upload audio*

> You can either record directly using your microphone, or upload an existing audio file.
>
> We support WAV, MP3, M4A, WebM, and OGG formats.

**[4:00-4:15]** *Show analysis running*

> We're not doing simple speech-to-text.
> We're analyzing the physics of the sound wave itself.
>
> Watch the pipeline: Uploading, Preprocessing, Extracting Biomarkers, Generating Assessment.

**[4:15-4:30]** *Results appear with biomarker cards*

> We extract 9 distinct biomarkers:
> Jitter, Shimmer, Harmonics-to-Noise Ratio, CPPS, Fundamental Frequency, Speech Rate, Formants, MFCCs, and Pause Patterns.
>
> Each compared against clinical baselines.

**[4:30-4:45]** *Show radar chart and risk score*

> The radar chart visualizes all biomarkers at once.
> And we generate a neurological risk score from 0 to 100.
>
> This transforms a smartphone microphone into a diagnostic medical device.

---

## SCENE 6: Cardiology Pipeline Demo (4:45 - 5:45)

**SCREEN:** CardioPredict AI - localhost:3000/dashboard/cardiology

**ACTION:** Navigate to Cardiology page, upload ECG, run analysis

**[4:45-5:00]** *Navigate to Cardiology page*

> Our third pillar: **CardioPredict AI**.
>
> Signal processing at its finest. We're dealing with high-frequency time-series data from Electrocardiograms.

**[5:00-5:15]** *Upload ECG file or use demo*

> You can upload raw ECG data or use our demo mode for instant results.
>
> The system handles 12-lead signal arrays with aggressive noise filtering.

**[5:15-5:30]** *Show analysis running*

> We filter power-line interference, remove baseline wander, and run specialized R-peak detection.
>
> By measuring distances between R-peaks, we calculate Heart Rate Variability, a powerful proxy for autonomic nervous system health.

**[5:30-5:45]** *Show results with rhythm classification*

> The output isn't just "Normal" or "Abnormal."
> We classify the specific rhythm type: Sinus Rhythm, Atrial Fibrillation, Bradycardia, Tachycardia.
>
> We extract 15+ biomarkers and return the exact time intervals that triggered any alerts.

---

## SCENE 7: AI Explanation + Voice (5:45 - 6:30)

**SCREEN:** Any pipeline results page - focus on ExplanationPanel

**ACTION:** Show AI explanation generating, then click voice button

**[5:45-6:00]** *Point to AI Explanation panel*

> Now watch this. After every analysis, our AI Explanation panel automatically generates.
>
> This is powered by **Cerebras Cloud** running **Llama 3.3 70B**, one of the fastest LLM inference engines available.

**[6:00-6:15]** *Show streaming text*

> The explanation streams in real-time, token by token.
> It synthesizes the biomarker results into plain English clinical context.
>
> "Based on the analysis, the patient shows elevated jitter values suggesting potential vocal instability..."

**[6:15-6:30]** *Click voice button, show audio playing*

> And here's the magic. Click this speaker icon.
>
> We integrated **AWS Polly** for text-to-speech. The AI explanation is now being read aloud.
>
> This is perfect for busy clinicians who want to listen while reviewing other data.

---

## SCENE 8: Architecture Overview (6:30 - 7:15)

**SCREEN:** GitHub - ARCHITECTURE.md file

**ACTION:** Open GitHub repo, navigate to ARCHITECTURE.md, scroll through diagrams

**[6:30-6:45]** *Open GitHub ARCHITECTURE.md*

> Let me show you the architecture, because that's where the real innovation lies.
>
> We've documented everything in this ARCHITECTURE.md file on GitHub.

**[6:45-7:00]** *Scroll to System Overview diagram*

> At the top level: Next.js frontend on Vercel, FastAPI backend on HuggingFace Spaces, connected to Cerebras Cloud and AWS Polly.
>
> Every component is decoupled and scalable.

**[7:00-7:15]** *Scroll through pipeline diagrams*

> Each pipeline follows a standardized layered architecture:
> Input validation, preprocessing, feature extraction, analysis, clinical assessment, and explainability.
>
> This modularity is what allows us to add new modalities without breaking existing ones.

---

## SCENE 9: AI Chatbot Demo (7:15 - 7:45)

**SCREEN:** Dashboard with Medical Chatbot open

**ACTION:** Click chatbot button, ask a medical question

**[7:15-7:30]** *Click chatbot icon in bottom right*

> We also built a context-aware Medical Chatbot.
>
> This isn't just GPT in a box. It has access to your assessment history and can synthesize across modalities.

**[7:30-7:45]** *Type a question like "What's my overall health profile?"*

> When you ask "What's my overall risk profile?", the chatbot pulls structured results from all your previous assessments.
>
> It connects the dots: vascular risk from the eye, neurological risk from voice, cardiac risk from ECG.
>
> This is true multimodal AI integration.

---

## SCENE 10: Additional Features (7:45 - 8:30)

**SCREEN:** Dashboard - tour various features

**ACTION:** Show sidebar, analytics, settings

**[7:45-8:00]** *Show sidebar navigation*

> Let me give you a quick tour of additional features.
>
> The sidebar shows all diagnostic modules, coming soon features, and quick access to Analytics and Reports.

**[8:00-8:15]** *Navigate to Analytics page*

> The Analytics page shows assessment trends, risk score distribution, and key metrics over time.
>
> This gives clinicians a longitudinal view of patient health.

**[8:15-8:30]** *Show responsive design, mobile preview*

> Everything is fully responsive. Whether you're on a desktop in a hospital, a tablet in a clinic, or a phone in the field... MediLens adapts.
>
> We also have HIPAA compliant indicators everywhere, real-time system status, and processed via Cerebras badges for transparency.

---

## SCENE 11: Coming Soon & Roadmap (8:30 - 9:15)

**SCREEN:** Dashboard sidebar showing Coming Soon section

**ACTION:** Scroll sidebar, show coming soon modules, navigate to Vision page

**[8:30-8:45]** *Show Coming Soon section in sidebar*

> Beyond our 9 live modules, we've roadmapped 5 additional Coming Soon modules:
>
> HistoVision AI for pathology, NeuroScan AI for brain imaging, RespiRate AI for respiratory analysis, FootCare AI for diabetic foot, and BoneScan AI for orthopedics.

**[8:45-9:00]** *Navigate to About or Vision page*

> Our vision? Democratizing AI-Powered Healthcare.
>
> We want advanced diagnostics accessible to every provider, regardless of resources or location.

**[9:00-9:15]** *Show roadmap section*

> By 2027, we aim to reach 1 million patients, deploy to 50+ countries, and partner with 100+ hospitals.
>
> This isn't a hackathon prototype. It's a roadmap to global health equity.

---

## SCENE 12: Closing & Impact (9:15 - 10:00)

**SCREEN:** Homepage hero, then fade to logo

**ACTION:** Return to homepage, slow scroll, fade to logo

**[9:15-9:30]** *Return to homepage*

> Imagine a nurse in a remote village in India. A mobile health van in rural America. A community clinic in sub-Saharan Africa.
>
> They don't have a retinologist. They don't have a neurologist. They don't have a cardiologist.

**[9:30-9:45]** *Slow zoom on logo*

> But armed with a smartphone and MediLens, they have the diagnostic power of a research hospital in their pocket.
>
> MediLens is not replacing doctors. We're supercharging triage so the sickest patients see specialists faster.

**[9:45-10:00]** *Fade to black with logo centered*

> We believe high-quality diagnostics should be a fundamental right, not a geographic privilege.
>
> This is **MediLens**.
> Production-ready. Scientifically grounded. Designed to scale to millions.
>
> Thank you.

---

## Recording Notes

### Browser Setup
- Chrome or Firefox in full-screen mode
- Resolution: 1920x1080
- Clear browser history/cache
- Log in to Clerk before recording
- Preload sample images/audio

### Sample Data Required
1. Fundus image (high quality, diabetic retinopathy example)
2. Audio file (30 second voice sample)
3. ECG data (demo mode sufficient)

### Post-Production
- Add subtle background music (ambient, medical tech feel)
- Add lower-third text for key features
- Add transition effects between scenes
- Color grade for consistency

---

*Script Version: 2.0 | Last Updated: January 2026*
