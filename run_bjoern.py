import bjoern
from app import app

# Bind to the PORT environment variable for Heroku
import os
port = int(os.environ.get('PORT', 5000))
