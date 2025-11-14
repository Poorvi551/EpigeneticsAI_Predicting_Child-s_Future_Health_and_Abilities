import json
import random
import os
import tempfile
from difflib import get_close_matches
from typing import List, Optional

# Voice feedback - Using gTTS
VOICE_AVAILABLE = False
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    VOICE_AVAILABLE = True
except ImportError:
    pass

# Optional emotion detection
EMOTION_DETECTION_AVAILABLE = False
try:
    import cv2
    from fer import FER
    EMOTION_DETECTION_AVAILABLE = True
except ImportError:
    pass

# Intent keywords
INTENTS = {
    "chat": ["hello", "hi", "how are you", "what's up", "hey", "greeting", "chat"],
    "study": ["study", "homework", "exam", "math", "science", "history", "coding", "learn", "revision"],
    "health": ["health", "doctor", "symptom", "mental", "wellbeing", "anxiety", "stress", "fitness"],
    "mood": ["mood", "happy", "sad", "booster", "anxious", "bored", "uplift", "calm"],
    "motivation": ["motivation", "focus", "lazy", "encouragement", "goal", "energy", "discipline"]
}

# Sentiment triggers
SENTIMENT_TRIGGERS = {
    "mood": ["i feel low", "i'm sad", "feeling down", "depressed", "anxious", "overwhelmed"],
    "motivation": ["i feel lazy", "can't focus", "unmotivated", "procrastinating", "no energy"]
}

FUN_FACTS = [
    "Did you know? Honey never spoils. Archaeologists found pots of it in ancient tombs!",
    "Octopuses have three hearts and blue blood. Wild, right?",
    "The Eiffel Tower can grow over 6 inches in summer due to heat expansion.",
    "Bananas are berries, but strawberries aren't. Nature loves plot twists!"
]

# Path to knowledge base
KB_PATH = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')

# Voice feedback function
def speak(text: str):
    """Convert text to speech using Google TTS (requires internet)"""
    if not VOICE_AVAILABLE:
        return
    
    try:
        # Clean text for speech
        clean_text = text.replace('💤', '').replace('👀', '').replace('💙', '').replace('🎉', '').replace('👶', '').replace('🤖', '').replace('✅', '').strip()
        
        if not clean_text:
            return
            
        print(f"🔊 Speaking: {clean_text[:50]}...")
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
        
        # Generate speech
        tts = gTTS(text=clean_text, lang='en', slow=False)
        tts.save(temp_file)
        
        # Play the audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Cleanup
        pygame.mixer.music.unload()
        os.remove(temp_file)
        
        print("✅ Finished speaking")
        
    except Exception as e:
        print(f"[Voice Error: {e}]")

# Emotion detection function
def detect_emotion_from_webcam() -> tuple[Optional[str], float]:
    """Detect emotion from webcam. Returns (emotion, confidence)"""
    if not EMOTION_DETECTION_AVAILABLE:
        return None, 0.0
    
    try:
        print("📸 Opening webcam for emotion detection... Please look at the camera!")
        detector = FER(mtcnn=False)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not access webcam")
            return None, 0.0

        emotion_detected = None
        confidence_score = 0.0
        frames_checked = 0
        max_frames = 30  # Check for 30 frames

        while frames_checked < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.detect_emotions(frame)
            if results:
                emotions = results[0].get("emotions", {})
                if emotions:
                    top_emotion = max(emotions, key=emotions.get)
                    top_score = float(emotions[top_emotion])
                    
                    # Update if we find a better confidence score
                    if top_score > confidence_score and top_score >= 0.5:
                        emotion_detected = top_emotion
                        confidence_score = top_score
                    
                    # Display emotion on frame
                    cv2.putText(frame, f"{top_emotion}: {top_score:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Emotion Detection - Press 'q' to finish", frame)
            frames_checked += 1
            
            # Allow early exit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        return emotion_detected, confidence_score
        
    except Exception as e:
        print(f"❌ Emotion detection error: {e}")
        return None, 0.0

# Load knowledge base
def load_knowledge_base(file_path: str = KB_PATH) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if "questions" not in data:
                data["questions"] = []
            return data
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
        return {"questions": []}

# Save knowledge base
def save_knowledge_base(data: dict, file_path: str = KB_PATH):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

# FIXED: Better matching function that checks main question AND similar phrases
def find_best_match(user_question: str, knowledge_base: dict) -> Optional[str]:
    """Find best match by checking both main questions and similar phrases"""
    user_lower = user_question.lower().strip()
    
    # First, try exact matching (case-insensitive)
    for q in knowledge_base["questions"]:
        # Check main question
        if q["question"].lower().strip() == user_lower:
            return q["question"]
        
        # Check similar phrases
        for similar in q.get("similar", []):
            if similar.lower().strip() == user_lower:
                return q["question"]
    
    # Build list of all possible matches (main questions + similar phrases)
    all_phrases = []
    phrase_to_question = {}  # Map phrases back to main question
    
    for q in knowledge_base["questions"]:
        main_q = q["question"]
        all_phrases.append(main_q)
        phrase_to_question[main_q] = main_q
        
        # Add similar phrases
        for similar in q.get("similar", []):
            all_phrases.append(similar)
            phrase_to_question[similar] = main_q
    
    # Use fuzzy matching with LOWER cutoff (0.4 instead of 0.6)
    matches = get_close_matches(user_question, all_phrases, n=1, cutoff=0.6)
    
    if matches:
        matched_phrase = matches[0]
        return phrase_to_question[matched_phrase]
    
    return None

# Get random answer
def get_random_answer(question: str, knowledge_base: dict) -> Optional[str]:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return random.choice(q["answers"])

# Add or update answer
def add_or_update_answer(user_question: str, new_answer: str, similar_phrases: List[str], intents: List[str], knowledge_base: dict):
    for q in knowledge_base["questions"]:
        if q["question"] == user_question:
            if new_answer not in q["answers"]:
                q["answers"].append(new_answer)
            q["intents"] = list(set(q["intents"] + intents))
            # Add similar phrases if not already present
            existing_similar = set(q.get("similar", []))
            for phrase in similar_phrases:
                if phrase and phrase not in existing_similar:
                    if "similar" not in q:
                        q["similar"] = []
                    q["similar"].append(phrase)
            return
    knowledge_base["questions"].append({
        "question": user_question,
        "similar": similar_phrases,
        "answers": [new_answer],
        "intents": intents
    })

# Detect intent
def detect_intent(user_input: str) -> Optional[str]:
    for intent, triggers in SENTIMENT_TRIGGERS.items():
        if any(trigger in user_input.lower() for trigger in triggers):
            return intent
    for intent, keywords in INTENTS.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            return intent
    return "chat"

# Optional fallback for fun fact
def get_fun_fact() -> str:
    return random.choice(FUN_FACTS)

# Main chatbot function with voice and emotion
def chat_bot():
    knowledge_base = load_knowledge_base()
    
    welcome_msg = "Hey Kidoo! I'm your study buddy, mood booster, and motivational pal. Type 'quit' to exit."
    print(f"🤖 {welcome_msg}\n")
    
    if VOICE_AVAILABLE:
        speak(welcome_msg)
    else:
        print("⚠️ Voice not available. Install with: pip install gtts pygame\n")

    # Detect emotion at startup
    emotion_tag = "neutral"
    if EMOTION_DETECTION_AVAILABLE:
        emotion, confidence = detect_emotion_from_webcam()
        if emotion:
            emotion_tag = emotion
            emotion_greetings = {
                "happy": "You're looking happy today! That's awesome!",
                "sad": "I sense you're feeling a bit low. I'm here to help cheer you up!",
                "angry": "Feeling fired up? Let's channel that energy positively!",
                "neutral": "Hey there! Ready to chat?",
                "surprised": "Whoa! You seem surprised! What's up?"
            }
            greeting = emotion_greetings.get(emotion, f"I detected you're feeling {emotion}")
            greeting_with_conf = f"{greeting} (confidence: {confidence:.2f})"
            print(f"😊 Bot: {greeting_with_conf}")
            speak(greeting)
        else:
            print("😊 Bot: I couldn't detect your emotion, but that's okay! Let's chat anyway.\n")
    else:
        print("⚠️ Emotion detection not available. Install with: pip install opencv-python fer\n")

    # Main chat loop
    while True:
        user_input = input('You: ').strip()

        if user_input.lower() == 'quit':
            goodbye = "Catch you later! Keep building magic"
            print(f"Bot: {goodbye} 💙")
            speak(goodbye)
            break

        intent = detect_intent(user_input)
        best_match = find_best_match(user_input, knowledge_base)

        if best_match:
            answer = get_random_answer(best_match, knowledge_base)
            print(f'Bot ({intent.capitalize()} | {emotion_tag.capitalize()}): {answer}')
            speak(answer)
        else:
            no_answer = "I don't know the answer. Can you teach me?"
            print(f"Bot ({intent.capitalize()} | {emotion_tag.capitalize()}): {no_answer}")
            speak(no_answer)
            
            new_answer = input('Type the answer or "skip" to skip: ').strip()

            if new_answer.lower() != 'skip':
                similar_input = input('Any similar phrases? (comma-separated or leave blank): ').strip()
                similar_phrases = [s.strip() for s in similar_input.split(',')] if similar_input else []
                add_or_update_answer(user_input, new_answer, similar_phrases, [intent], knowledge_base)
                save_knowledge_base(knowledge_base)
                learned = "Thanks! I've learned something new."
                print(f'✅ Bot: {learned}')
                speak(learned)
            else:
                skip_msg = "No worries! Ask me something else."
                print(f"Bot: {skip_msg}")
                speak(skip_msg)

# Helper function for Flask integration
def teach_bot(user_question: str, user_answer: str, kb_path: str = KB_PATH):
    """Teach the bot a new answer"""
    knowledge_base = load_knowledge_base(kb_path)
    intent = detect_intent(user_question)
    similar_phrases = []
    add_or_update_answer(user_question, user_answer, similar_phrases, [intent], knowledge_base)
    save_knowledge_base(knowledge_base, kb_path)
    return "Thanks for teaching me! I'll remember that!"

# Get bot reply for Flask integration
def get_bot_reply(user_input: str, kb_path: str = KB_PATH) -> str:
    """Get bot reply for web interface"""
    knowledge_base = load_knowledge_base(kb_path)

    intent = detect_intent(user_input)
    best_match = find_best_match(user_input, knowledge_base)

    if best_match:
        answer = get_random_answer(best_match, knowledge_base)
        return answer
    else:
        return "I don't know the answer to that yet. Can you teach me? Type your answer or say 'skip'."

if __name__ == "__main__":
    chat_bot()