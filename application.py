from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Single-transaction prediction page."""
    if request.method == "GET":
        return render_template("predict.html")

    # POST: form submission
    try:
        # Read form fields
        step = int(request.form.get("step", 1))
        type_ = request.form.get("type", "TRANSFER")
        amount = float(request.form.get("amount", 0.0))
        nameOrig = request.form.get("nameOrig", "C123456789")
        oldbalanceOrg = float(request.form.get("oldbalanceOrg", 0.0))
        newbalanceOrig = float(request.form.get("newbalanceOrig", 0.0))
        nameDest = request.form.get("nameDest", "C987654321")
        oldbalanceDest = float(request.form.get("oldbalanceDest", 0.0))
        newbalanceDest = float(request.form.get("newbalanceDest", 0.0))
        isFlaggedFraud = int(request.form.get("isFlaggedFraud", 0))

        # Wrap into CustomData
        data = CustomData(
            step=step,
            type=type_,
            amount=amount,
            nameOrig=nameOrig,
            oldbalanceOrg=oldbalanceOrg,
            newbalanceOrig=newbalanceOrig,
            nameDest=nameDest,
            oldbalanceDest=oldbalanceDest,
            newbalanceDest=newbalanceDest,
            isFlaggedFraud=isFlaggedFraud,
        )

        df = data.to_dataframe()
        pipeline = PredictPipeline()
        pred = int(pipeline.predict(df)[0])

        result_text = "FRAUD" if pred == 1 else "LEGITIMATE"

        return render_template("predict.html", prediction=result_text, form_data=request.form)

    except Exception as e:
        # Simple error display; logging already handled in backend
        return render_template("predict.html", error=str(e), form_data=request.form)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
