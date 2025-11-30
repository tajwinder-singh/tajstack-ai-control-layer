from flask import Flask, render_template, request

# This is a demo.
# No retrieval, no knowledge base, no multi-tenant logic.
# The real production logic is private.

from main_logic import simple_generate_email

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    purpose = request.form.get("purpose", "")
    tone = request.form.get("tone", "Formal")
    key_points = request.form.get("key_points", "")

    # Call safe demo generator
    output = simple_generate_email(purpose, tone, key_points)

    return render_template("result.html", output=output)


if __name__ == "__main__":
    app.run(debug=True)
