from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        if username == "U101" and password == "123":
            # return 'Login successful!'
            return redirect(url_for("home"))
    return render_template('login.html')

@app.route('/signin')
def signin():
    return render_template("signin.html")

if __name__=="__main__":
    app.run(port=5000, debug=True)