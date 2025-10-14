"""
Docker-specific entry point for SLP SimuCase Generator

This file imports create_app() from main.py to ensure 100% consistency.
The ONLY difference is the launch() configuration for Docker.

Any changes to main.py automatically apply here!
"""

# Import the app creation function from main.py
from main import create_app

if __name__ == "__main__":
    app = create_app()

    # Docker configuration: Use 0.0.0.0 to be accessible from outside container
    # Port 7860 is exposed and mapped in docker-compose.yml
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
