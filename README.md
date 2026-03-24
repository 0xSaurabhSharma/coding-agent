# 1. Initialize the project (creates pyproject.toml and a virtual environment)
uv init

# 2. Set the Python version to 3.12
uv python pin 3.12

# 3. Add the project dependency (automatically handles install and updates pyproject.toml)
uv add openrouter

# 4. Initialize Git
git init

# 5. Setup Git ignore
echo ".venv/" > .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".python-version" >> .gitignore

# 6. Create environment variable template
echo "OPENROUTER_API_KEY=your_actual_key_here" > .env.example