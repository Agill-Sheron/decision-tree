import pandas as pd
import graphviz
from flask import Flask, render_template, request, send_file, jsonify
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import json


app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload-data', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        global df, columns
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        return columns


@app.route('/build-decision-tree', methods=['POST'])
def build_decision_tree_route():
    # Get the target column from the request
    target_column = json.loads(request.data)['target_column']

    # Check if the target column is in the dataframe
    if target_column not in df.columns:
        return jsonify({'error': 'Invalid target column'}), 400

    # Set the target_values variable
    target_values = df[target_column]

    # Drop the target column from the dataframe
    df_features = df.drop(target_column, axis=1)

    # Convert categorical variable into dummy/indicator variables
    df_features = pd.get_dummies(df_features)

    # Get the list of feature names
    feature_names = list(df_features.columns)

    # Build the decision tree using sklearn's DecisionTreeClassifier
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(df_features, target_values)

    # Render the decision tree in the frontend
    dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                    feature_names=feature_names,
                                    class_names=['Will not wait', 'Will wait'],
                                    filled=True)
    graph = graphviz.Source(dot_data, format="png")
    graph.render("static/decision_tree")

    return send_file("static/decision_tree.png", mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
