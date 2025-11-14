import json
import os

KB_PATH = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base.json')

def load_kb():
    if not os.path.exists(KB_PATH):
        return {"questions": []}
    with open(KB_PATH, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_kb(data):
    with open(KB_PATH, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

def update_knowledge_base(question, answer, similar_phrases, intents, kb_path=KB_PATH):
    kb = load_kb()

    # Normalize question for matching
    question_lower = question.lower()
    for q in kb["questions"]:
        if q["question"].lower() == question_lower:
            if answer not in q["answers"]:
                q["answers"].append(answer)
            q["similar"] = list(set(q["similar"] + similar_phrases))
            q["intents"] = list(set(q["intents"] + intents))
            save_kb(kb)
            print(f"✅ Updated existing entry: {question}")
            return

    # Add new entry
    kb["questions"].append({
        "question": question,
        "similar": similar_phrases,
        "answers": [answer],
        "intents": intents
    })
    save_kb(kb)
    print(f"✅ Added new entry: {question}")

def add_entries_to_knowledge_base():
    new_entries = [
        {
            "question": "how to concentrate while studying",
            "similar": ["how can I focus", "I get distracted while studying", "can't concentrate"],
            "answers": [
                "Try studying in short bursts with breaks in between—it's called the Pomodoro technique!",
                "Find a quiet space, keep your phone away, and set a small goal for each session.",
                "Start with subjects you enjoy—it builds momentum and makes it easier to tackle harder ones."
            ],
            "intents": ["study"]
        },
        {
            "question": "I'm bored and not in the mood to study",
            "similar": ["I don't feel like studying", "I'm tired of studying", "I can't focus today"],
            "answers": [
                "Even superheroes take breaks—but they always come back stronger. Let’s start with just 5 minutes!",
                "Studying isn’t about being perfect—it’s about showing up. You’ve got this!",
                "Let’s make it fun! Try using colors, flashcards, or teaching someone else what you learned.",
                "Your dreams are waiting for you. Every small step today brings you closer to them.",
                "You’re not alone. I believe in you—and I’m here to help you through it."
            ],
            "intents": ["motivation"]
        },
        {
            "question": "I feel sad",
            "similar": ["I'm upset", "I'm feeling low", "I'm not happy"],
            "answers": [
                "I'm here for you. Want to talk about what’s bothering you?",
                "It’s okay to feel sad sometimes. You're strong, and this feeling will pass.",
                "Even the brightest stars have cloudy nights. You’re still shining."
            ],
            "intents": ["mood"]
        },
        {
            "question": "I failed my test",
            "similar": ["I didn't do well", "I got bad marks", "I'm disappointed"],
            "answers": [
                "One test doesn’t define you. You’ve got so much potential!",
                "Failure is just feedback. Let’s learn from it and come back stronger.",
                "You tried, and that matters. Let’s figure out what went wrong and fix it together."
            ],
            "intents": ["motivation"]
        },
        {
            "question": "how to remember what I study",
            "similar": ["I forget everything", "how to improve memory", "can't retain information"],
            "answers": [
                "Teach someone else what you learned—it’s the best way to remember!",
                "Use mind maps or flashcards to make studying fun and visual.",
                "Review your notes regularly instead of cramming. Your brain loves repetition!"
            ],
            "intents": ["study"]
        }
    ]

    kb = load_kb()
    existing_questions = [q["question"].lower() for q in kb["questions"]]

    for entry in new_entries:
        if entry["question"].lower() not in existing_questions:
            kb["questions"].append(entry)

    save_kb(kb)
    print("✅ Motivational and educational entries added to knowledge_base.json")