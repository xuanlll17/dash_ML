from flask import Flask, redirect
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from dash_file.dash_ML import dash_ML


app = Flask(__name__)
application = DispatcherMiddleware(
    app,
    {
     "/dash/ML": dash_ML.server
    },
)

@app.route("/")
def index():
    return redirect('/dash/ML')

if __name__ == "__main__":
    run_simple("localhost", 8080, application,use_debugger=True,use_reloader=True)