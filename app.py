from flask import Flask, render_template, request

# Public interaction surface only.
# Production control logic is private.

from main_logic import generate_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/trigger", methods=["POST"])
def trigger():
    request_context = request.form.get("request_context", "")
    response_mode = request.form.get("response_mode", "Formal")
    operational_notes = request.form.get("operational_notes", "")

    output = generate_response(request_context, response_mode, operational_notes)

    return render_template("result.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)
