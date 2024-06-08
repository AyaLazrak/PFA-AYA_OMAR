from flask import Blueprint

auth = Blueprint('auth',__name__) #define the Blueprint

@auth.route('/login')
def login():
    return "<p>Login</p>"

@auth.route('/logout')
def logout():
    return "<p>Logout</p>"

@auth.route('/sign_up')
def sign_up():
    return "<p>sign-up</p>"
