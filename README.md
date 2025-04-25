# Real-time RSI Monitor & Notifier

This service monitors real-time market data from Bybit (Linear Perpetuals), calculates the Relative Strength Index (RSI) for USDT pairs, and sends push notifications via Expo when RSI values cross predefined thresholds (overbought/oversold).

## Features

*   Connects to Bybit WebSocket for real-time Kline data.
*   Calculates RSI for all active USDT perpetual contracts.
*   Fetches Expo push tokens from a Supabase database.
*   Sends push notifications using the Expo Push API when RSI thresholds are met.
*   Configurable RSI period, thresholds, Kline interval, and alert cooldown via environment variables.
*   Built with FastAPI, designed for deployment on services like Render.
*   Includes basic status endpoint (`/status`).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory for local development (this file should **not** be committed to Git). Add the following variables:

    ```dotenv
    # Bybit API Credentials (Required)
    BYBIT_API_KEY="YOUR_BYBIT_API_KEY"
    BYBIT_API_SECRET="YOUR_BYBIT_API_SECRET"

    # Supabase Credentials (Required)
    SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL"
    # Use the SERVICE_ROLE key here for backend access
    SUPABASE_SERVICE_KEY="YOUR_SUPABASE_SERVICE_ROLE_KEY"

    # --- Optional Configuration ---

    # RSI Calculation
    # RSI_PERIODS=14
    # OVERBOUGHT_THRESHOLD=90
    # OVERSOLD_THRESHOLD=10
    # KLINE_INTERVAL=240 # e.g., 60 (1H), 240 (4H), D (Daily)

    # Alerting
    # ALERT_COOLDOWN_SECONDS=300 # 5 minutes
    ```

    When deploying (e.g., to Render), set these as environment variables directly in the service configuration.

## Running Locally

Use Uvicorn to run the FastAPI application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*   `--reload` enables auto-reloading during development.
*   Access the service at `http://localhost:8000`.
*   Check the status at `http://localhost:8000/status`.

## Deployment (Render Example)

1.  Push your code to a Git repository (GitHub, GitLab).
2.  Create a new "Web Service" on Render.
3.  Connect Render to your Git repository.
4.  Set the **Build Command** to: `pip install -r requirements.txt`
5.  Set the **Start Command** to: `uvicorn main:app --host 0.0.0.0 --port $PORT` (Render sets the `$PORT` variable automatically).
6.  Go to the "Environment" section for your service.
7.  Add all the required environment variables (`BYBIT_API_KEY`, `BYBIT_API_SECRET`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`) and any optional ones you want to override.
8.  Deploy the service.
9.  Monitor the logs on Render for status updates and potential errors.

## Supabase Setup Reminder

Ensure you have a `push_tokens` table in your Supabase database with at least the following columns, as per `rsi_notification_plan.md`:

*   `user_id` (UUID, FK to `auth.users.id`)
*   `token` (TEXT, should store the ExpoPushToken)
*   Enable Row Level Security (RLS) for secure access from the frontend app, but ensure the backend uses the `service_role` key to bypass RLS for reading all tokens when sending notifications.