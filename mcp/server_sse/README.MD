# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
.venv\Scripts\activate

# Install dependencies
uv add mcp[cli] httpx

# Create our server file
new-item weather.py

# Run Server
npx -y @modelcontextprotocol/inspector
uv run src\weather.py

Transport Type: SSE
URL: http://0.0.0.0:8000/sse