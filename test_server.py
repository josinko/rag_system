# test_server.py - Minimal test to check if Flask works

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/test')
def test():
    return jsonify({"status": "success", "message": "Test successful"})

if __name__ == '__main__':
    print("=" * 50)
    print("Starting test Flask server...")
    print("If you see this, Python is running the script")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting Flask: {e}")
        input("Press Enter to exit...")