from flask import Flask, render_template, request, jsonify
from model_wrapper import ResumeModel

app = Flask(__name__)

# Load model once (IMPORTANT)
resume_model = ResumeModel()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    resume_text = data.get("resume", "")
    career_text = data.get("career", "")
    jd_text = data.get("job", "")

    if not resume_text or not jd_text:
        return jsonify({"error": "Missing input"}), 400

    score = resume_model.predict_score(
        resume_text,
        career_text,
        jd_text
    )

    percentage = round(score * 100, 2)

    if percentage >= 80:
        verdict = "Excellent Fit"
    elif percentage >= 60:
        verdict = "Good Fit"
    elif percentage >= 40:
        verdict = "Average Fit"
    else:
        verdict = "Low Match"

    return jsonify({
        "score": percentage,
        "verdict": verdict
    })

if __name__ == "__main__":
    app.run(debug=True)
