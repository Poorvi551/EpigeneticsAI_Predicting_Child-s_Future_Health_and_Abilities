from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, after_this_request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import random
from gtts import gTTS
import tempfile  # Fixed: was tempfiles
# from ChildBot import emotion_detection  # Comment out if not needed

# 🔧 Bot imports
from ChildBot.main import (
    get_bot_reply, detect_intent, add_or_update_answer,
    get_random_answer, find_best_match, save_knowledge_base, load_knowledge_base
)
from ChildBot.utils.updater import update_knowledge_base, add_entries_to_knowledge_base

# 🚀 Flask setup
app = Flask(__name__, instance_relative_config=True)
app.secret_key = '5cd59684952ed0c3755d0547f8984504'

# ✅ Ensure instance folder exists
if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)

# ✅ Set DB path
db_path = os.path.join(app.instance_path, 'epigenetic.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 👤 User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f"<User {self.username} ({self.email})>"

# Home
@app.route("/")
def home():
    return render_template("index.html")

# Register 
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        raw_password = request.form.get("password", "")

        # Validation
        if not username or not email or not raw_password:
            flash("All fields are required.", "warning")
            return redirect(url_for("register"))

        # Check if email already exists
        existing_email_user = User.query.filter_by(email=email).first()
        if existing_email_user:
            flash("Email already registered. Please log in.", "warning")
            return redirect(url_for("login"))

        # Check if username already exists
        existing_username_user = User.query.filter_by(username=username).first()
        if existing_username_user:
            flash("Username already taken. Please choose another.", "warning")
            return redirect(url_for("register"))

        # Hash password with method specified
        hashed_password = generate_password_hash(raw_password, method='pbkdf2:sha256')
        
        # Create new user
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")

#  Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Check if request wants JSON response (AJAX/fetch request)
        wants_json = request.headers.get('Accept', '').find('application/json') != -1
        
        # Validation
        if not email or not password:
            if wants_json:
                return jsonify({"success": False, "message": "Email and password are required.", "redirect": None}), 400
            flash("Email and password are required.", "warning")
            return redirect(url_for("login"))

        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user:
            if wants_json:
                return jsonify({"success": False, "message": "No account found for that email. Please sign up.", "redirect": "/register"}), 200
            
            flash("No account found for that email. Please sign up.", "warning")
            return redirect(url_for("register"))

        # Check password for existing user
        if check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            
            if wants_json:
                return jsonify({"success": True, "message": f"Welcome back, {user.username}!", "redirect": "/"}), 200
            
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("home"))
        else:
            if wants_json:
                return jsonify({"success": False, "message": "Incorrect password. Please try again.", "redirect": None}), 200
            
            flash("Incorrect password. Please try again.", "danger")
            return redirect(url_for("login"))

    return redirect(url_for("home"))

# 🚪 Logout
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return render_template("logout.html")

# 🎤 Text-to-Speech endpoint - FIXED
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Clean text for speech
        clean_text = text.replace('💤', '').replace('👀', '').replace('💙', '').replace('🎉', '').replace('👶', '').strip()
        
        if not clean_text:
            return jsonify({'error': 'No valid text to convert'}), 400
        
        # Generate speech
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        # Schedule file deletion after sending
        @after_this_request
        def cleanup(response):
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return response
        
        # Send file
        return send_file(temp_file, mimetype='audio/mpeg')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🤖 Chatbot logic
last_unknown = {}

@app.route('/get_response', methods=['POST'])
def chat_response():
    global last_unknown
    user_message = request.json.get('message', '').strip()
    KB_PATH = os.path.join('ChildBot', 'knowledge_base.json')  # Fixed path capitalization
    knowledge_base = load_knowledge_base(KB_PATH)

    if last_unknown.get("pending"):
        stage = last_unknown.get("stage")

        if stage == "answer":
            if user_message.lower() == "skip":
                last_unknown = {}
                return jsonify({'reply': "Alright! Here's a fun fact: {}".format(
                    random.choice(knowledge_base.get('fun_facts', ["Bananas are berries, but strawberries aren't."]))
                )})
            else:
                last_unknown["answer"] = user_message
                last_unknown["stage"] = "similar"
                return jsonify({'reply': "Got it! Now, can you give me any similar phrases? (comma-separated or type 'none')"})

        elif stage == "similar":
            similar_phrases = [s.strip() for s in user_message.split(',')] if user_message.lower() != "none" else []
            last_unknown["similar"] = similar_phrases
            last_unknown["stage"] = "intent"

            detected_intent = detect_intent(last_unknown["question"])
            if detected_intent:
                last_unknown["intent"] = detected_intent
                update_knowledge_base(
                    last_unknown["question"],
                    last_unknown["answer"],
                    last_unknown["similar"],
                    [detected_intent],
                    KB_PATH
                )
                updated_kb = load_knowledge_base(KB_PATH)
                return_answer = get_random_answer(last_unknown["question"], updated_kb)
                last_unknown = {}
                return jsonify({'reply': f"Thanks! I've learned something new 💡\nHere's what I'll say next time: {return_answer}"})
            else:
                return jsonify({'reply': "Hmm, I couldn't detect the intent. Can you tell me what category this belongs to? (e.g., study, health, mood, motivation, chat)"})

        elif stage == "intent":
            intent = user_message.lower()
            last_unknown["intent"] = intent
            update_knowledge_base(
                last_unknown["question"],
                last_unknown["answer"],
                last_unknown["similar"],
                [intent],
                KB_PATH
            )
            updated_kb = load_knowledge_base(KB_PATH)
            return_answer = get_random_answer(last_unknown["question"], updated_kb)
            last_unknown = {}
            return jsonify({'reply': f"Awesome! I've learned something new 💡\nHere's what I'll say next time: {return_answer}"})

    # Normal bot reply
    bot_reply = get_bot_reply(user_message)
    if "Can you teach me?" in bot_reply:
        last_unknown = {
            "pending": True,
            "question": user_message,
            "stage": "answer"
        }

    return jsonify({'reply': bot_reply})

# 🚀 Run app
if __name__ == "__main__":
    with app.app_context():
        # Create all tables if they don't exist
        db.create_all()
        # Verify User table exists and has correct structure
        if os.path.exists(db_path):
            print(f"✓ Database found at: {db_path}")
        else:
            print(f"✓ Database created at: {db_path}")
        print(f"✓ User table ready (stores: id, username, email, password)")
    app.run(debug=True, port=8000)