from flask import Flask, render_template

app = Flask(__name__)

# Paste your Python code here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scatter_plot')
def scatter_plot():
    return render_template('scatter_plot.html')

@app.route('/correlation_matrix')
def correlation_matrix():
    return render_template('correlation_matrix.html')

# Add routes for other visualizations if needed

if __name__ == '__main__':
    app.run(debug=True)
