import joblib
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)


X = [[20], [30], [40], [50], [55], [65], [75]]
y = ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass']


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)


joblib.dump(model, "model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        num = float(request.form.get("number"))
        prediction = model.predict([[num]])[0]  # get value
    return render_template("index.html", pred=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
