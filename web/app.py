#!/usr/bin/env python3
"""
Flask web app for comparing algorithm allocations vs manual packing.

Usage:
    python3 web/app.py                    # dev server on localhost:5000
    python3 web/app.py --port 8080        # custom port
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request

from web.comparison import build_comparison_data, get_available_algorithms, get_available_offers

app = Flask(__name__)


@app.route("/")
def index():
    offers = get_available_offers()
    algorithms = get_available_algorithms()
    return render_template("index.html", offers=offers, algorithms=algorithms)


@app.route("/compare")
def compare():
    offer_id = request.args.get("offer", type=int)
    algorithm = request.args.get("algorithm", default="ilp-optimal")

    if offer_id is None:
        return render_template("index.html",
                               offers=get_available_offers(),
                               algorithms=get_available_algorithms(),
                               error="Please select an offer.")

    try:
        data = build_comparison_data(offer_id, algorithm)
    except Exception as e:
        return render_template("index.html",
                               offers=get_available_offers(),
                               algorithms=get_available_algorithms(),
                               error=f"Error processing offer {offer_id}: {e}")

    return render_template("compare.html", data=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mystery Manager comparison web app")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)
