import asyncio
import os
import sys
import time
import logging
import threading
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pybit.unified_trading import HTTP, WebSocket
from supabase import create_client, Client
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# --- Configuration ---
# Bybit Keys (Required)
BYBIT_API_KEY = os.environ['BYBIT_API_KEY']
BYBIT_API_SECRET = os.environ['BYBIT_API_SECRET']

# Supabase Config (Required)
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_SERVICE_KEY = os.environ['SUPABASE_SERVICE_KEY']

# RSI Config
RSI_PERIODS = int(os.getenv('RSI_PERIODS', '14'))
OVERBOUGHT_THRESHOLD = int(os.getenv('OVERBOUGHT_THRESHOLD', '90'))
OVERSOLD_THRESHOLD = int(os.getenv('OVERSOLD_THRESHOLD', '10'))
KLINE_INTERVAL = os.getenv('KLINE_INTERVAL', '240') # Default to 4H

# Alerting Config
ALERT_COOLDOWN_SECONDS = int(os.getenv('ALERT_COOLDOWN_SECONDS', '300')) # 5 minutes
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

# WebSocket Config
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 15
HEARTBEAT_INTERVAL = 30 # Check connection every 30s
NO_MESSAGE_TIMEOUT = 120 # Reconnect if no messages for 120s
MAX_RECONNECT_DELAY = 300 # Max delay between reconnect attempts

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout for Render
)
logger = logging.getLogger(__name__)

# --- RSI Calculation ---
def calculate_rsi(prices, periods=RSI_PERIODS):
    """Calculate RSI from a pandas Series of prices"""
    if len(prices) < periods + 1: # Need at least periods + 1 data points
        return np.nan
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=periods - 1, adjust=False).mean() # Use com=periods-1 for standard RSI
    avg_loss = loss.ewm(com=periods - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] # Return the last calculated RSI value


# --- RSI Monitor Class ---
class RSIMonitor:
    def __init__(self, supabase_client: Client):
        logger.info("Initializing RSI Monitor...")
        self.supabase: Client = supabase_client
        self.session = HTTP(
            testnet=False, # Set to True for testing if needed
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        self.ws = None # WebSocket object, initialized later
        self.price_data = defaultdict(list)
        self.symbols = []
        self.subscribed_symbols = set()
        self.lock = threading.Lock()
        self.is_running = False
        self.rsi_values = {}
        self.last_alert_time = defaultdict(float)
        self.connected = False
        self.reconnect_count = 0
        self.last_message_time = 0
        self._stop_event = threading.Event()

    def _get_all_symbols(self):
        """Get list of actively trading USDT linear perpetual symbols"""
        logger.info("Fetching available symbols from Bybit...")
        try:
            max_retries = 3
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    instruments = self.session.get_instruments_info(
                        category="linear",
                        limit=1000
                    )
                    if instruments.get("retCode") != 0:
                        raise Exception(f"Bybit API error: {instruments.get('retMsg', 'Unknown error')}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to get instruments (attempt {attempt+1}/{max_retries}): {e}")
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"Failed to get instruments after {max_retries} attempts: {e}")
                        raise
            
            symbols = sorted([
                s["symbol"] for s in instruments["result"]["list"]
                if s["symbol"].endswith("USDT") and s["status"] == "Trading"
            ])
            logger.info(f"Found {len(symbols)} actively trading USDT linear perpetual symbols.")
            logger.info(f"First 5: {symbols[:5]}, Last 5: {symbols[-5:]}")
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols: {e}", exc_info=True)
            return []

    def init_websocket(self):
        """Initialize or reinitialize the WebSocket connection"""
        logger.info("Initializing WebSocket connection...")
        try:
            if self.ws:
                try:
                    # Attempt graceful shutdown if connection exists
                    self.ws.exit()
                    logger.info("Existing WebSocket connection closed.")
                except Exception as e:
                    logger.warning(f"Error closing existing WebSocket: {e}")

            self.ws = WebSocket(
                testnet=False,
                channel_type="linear",
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
                trace_logging=False # Set to True for debugging WS messages
            )
            self.connected = False # Mark as not connected until subscription succeeds
            self.last_message_time = time.time() # Reset message timer
            logger.info("WebSocket connection initialized.")
        except Exception as e:
            logger.error(f"WebSocket initialization error: {e}", exc_info=True)
            self.ws = None # Ensure ws is None if init fails


    def _heartbeat_monitor_thread(self):
        """Runs in a thread to monitor WebSocket connection health."""
        logger.info("Heartbeat monitor thread started.")
        while not self._stop_event.is_set():
            current_time = time.time()
            if self.ws and self.last_message_time > 0 and (current_time - self.last_message_time > NO_MESSAGE_TIMEOUT) and len(self.subscribed_symbols) > 0:
                logger.warning(f"No messages received for {NO_MESSAGE_TIMEOUT} seconds. Triggering reconnect.")
                self.connected = False
                # Reconnect will be handled by the main loop or another mechanism
                # For simplicity, just log here and let the subscription process handle it
                self.last_message_time = time.time() # Reset timer to avoid rapid reconnect triggers

            time.sleep(HEARTBEAT_INTERVAL)
        logger.info("Heartbeat monitor thread stopped.")

    def initialize_price_history(self):
        """Load historical price data for all symbols to bootstrap RSI calculation"""
        logger.info("Starting historical price data initialization...")
        if not self.symbols:
            self.symbols = self._get_all_symbols()
            if not self.symbols:
                logger.error("Cannot initialize price history: Failed to fetch symbols.")
                return False # Indicate failure

        with self.lock:
            self.price_data.clear()
            self.rsi_values.clear()

        total_symbols = len(self.symbols)
        for idx, symbol in enumerate(self.symbols, 1):
            if not self.is_running:
                logger.info("Stopping price history initialization.")
                return False

            logger.info(f"Initializing {symbol} ({idx}/{total_symbols})...")
            try:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        kline_data = self.session.get_kline(
                            category="linear",
                            symbol=symbol,
                            interval=KLINE_INTERVAL,
                            limit=RSI_PERIODS + 100 # Get enough data for initial RSI + buffer
                        )
                        if kline_data.get("retCode", -1) != 0:
                            msg = kline_data.get('retMsg', 'Unknown API error')
                            logger.warning(f"API error for {symbol}: {msg}")
                            if "rate limit" in msg.lower():
                                wait_time = 1 * (2 ** attempt)
                                logger.warning(f"Rate limit hit for {symbol}. Waiting {wait_time}s...")
                                time.sleep(wait_time)
                                continue # Retry
                            else:
                                # Break for non-rate-limit errors after logging
                                break
                        break # Success
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 1 * (2 ** attempt)
                            logger.warning(f"Failed to get kline for {symbol} (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to get kline for {symbol} after {max_retries} attempts.", exc_info=True)
                            raise # Reraise after max retries

                if kline_data.get("retCode") == 0 and kline_data.get("result", {}).get("list"):
                    prices = [float(k[4]) for k in kline_data["result"]["list"]]
                    prices.reverse() # API returns newest first

                    if len(prices) >= RSI_PERIODS + 1:
                        with self.lock:
                            self.price_data[symbol] = prices
                            rsi = calculate_rsi(pd.Series(prices))
                            if not np.isnan(rsi):
                                self.rsi_values[symbol] = rsi
                                logger.info(f"Initial RSI for {symbol}: {rsi:.2f}")
                                # Check initial threshold without cooldown
                                self._check_and_send_alert(symbol, rsi, is_initial=True)
                        logger.info(f"Loaded {len(prices)} historical prices for {symbol}. Initial RSI: {self.rsi_values.get(symbol, 'N/A')}")
                    else:
                        logger.warning(f"Insufficient historical data for {symbol} ({len(prices)} points). Need {RSI_PERIODS + 1}.")
                else:
                     logger.warning(f"Could not retrieve valid kline data for {symbol}. API Response: {kline_data}")

                time.sleep(0.2) # Avoid hitting rate limits

            except Exception as e:
                logger.error(f"Error initializing price history for {symbol}: {e}", exc_info=True)
                continue # Continue with the next symbol

        logger.info("Historical data initialization complete.")
        return True # Indicate success

    def subscribe_to_klines(self):
        """Subscribe to WebSocket kline streams for all symbols"""
        if not self.ws:
            logger.error("WebSocket not initialized. Cannot subscribe.")
            return False

        if not self.symbols:
            logger.warning("Symbols list is empty. Attempting to fetch.")
            self.symbols = self._get_all_symbols()
            if not self.symbols:
                logger.error("Cannot subscribe: Failed to fetch symbols.")
                return False

        logger.info("Starting WebSocket kline subscriptions...")

        def handle_message_wrapper(message):
            """Wrapper to update last message time and call main handler"""
            self.last_message_time = time.time()
            self.connected = True # Mark as connected once messages flow
            self._handle_kline_message(message)

        # Clear previous subscriptions before starting new ones
        self.subscribed_symbols.clear()
        subscription_args = [f"kline.{KLINE_INTERVAL}.{symbol}" for symbol in self.symbols]

        # Subscribe in batches to avoid potential issues with too many args at once
        batch_size = 50
        successfully_subscribed = set()
        for i in range(0, len(subscription_args), batch_size):
            batch = subscription_args[i:i+batch_size]
            try:
                self.ws.subscribe(batch, callback=handle_message_wrapper)
                # Extract symbols from the subscribed topics
                for topic in batch:
                    parts = topic.split('.')
                    if len(parts) == 3:
                        successfully_subscribed.add(parts[2])
                logger.info(f"Subscribed to batch {i//batch_size + 1}/{(len(subscription_args)+batch_size-1)//batch_size}, {len(successfully_subscribed)} symbols total.")
                time.sleep(1) # Small delay between batches
            except Exception as e:
                logger.error(f"Error subscribing to batch {i//batch_size + 1}: {e}")
                # Potentially retry or handle specific symbols failing

        self.subscribed_symbols = successfully_subscribed
        if not self.subscribed_symbols:
             logger.error("Failed to subscribe to any kline streams.")
             return False

        logger.info(f"Successfully subscribed to {len(self.subscribed_symbols)} kline streams.")
        self.last_message_time = time.time() # Reset timer after successful subscriptions
        self.connected = True
        self.reconnect_count = 0 # Reset reconnect counter on successful subscription
        return True

    def _handle_kline_message(self, message):
        """Process incoming kline WebSocket messages and update RSI values"""
        try:
            if "topic" not in message or "data" not in message or not message["data"]:
                # logger.debug(f"Ignoring non-kline or empty message: {message}")
                return

            topic = message["topic"]
            symbol = topic.split(".")[2] # Assumes topic format "kline.INTERVAL.SYMBOL"

            if symbol not in self.symbols:
                 logger.warning(f"Received message for unexpected symbol: {symbol}")
                 return # Ignore symbols we didn't intend to track

            data = message["data"][0]
            is_closed = data.get("confirm", False)
            current_price = float(data["close"])
            timestamp_ms = int(data["start"]) # Start time of the candle

            with self.lock:
                current_prices = self.price_data.get(symbol, [])
                new_rsi = np.nan

                if is_closed:
                    # Check if this candle is newer than the last recorded price
                    # This logic might need refinement based on how timestamps/updates work
                    if not current_prices or timestamp_ms > getattr(current_prices[-1], 'timestamp_ms', 0):
                         # Add the price (maybe store timestamp too if needed)
                         self.price_data[symbol].append(current_price)
                         # Keep buffer size reasonable
                         if len(self.price_data[symbol]) > RSI_PERIODS + 200:
                             self.price_data[symbol] = self.price_data[symbol][-(RSI_PERIODS + 200):]
                         logger.debug(f"CLOSED candle for {symbol}. Price: {current_price}. History length: {len(self.price_data[symbol])}")
                         # Recalculate RSI on closed candle
                         price_series = pd.Series(self.price_data[symbol])
                         new_rsi = calculate_rsi(price_series)

                else: # Unconfirmed (intermediate) candle update
                    if not current_prices:
                        # Should ideally not happen if history is initialized, but handle defensively
                        self.price_data[symbol].append(current_price)
                        logger.debug(f"First price point (unconfirmed) for {symbol}: {current_price}")
                    else:
                        # Create a temporary series with the latest price updated
                        temp_prices = current_prices[:-1] + [current_price]
                        price_series = pd.Series(temp_prices)
                        new_rsi = calculate_rsi(price_series)
                        # Optional: Log only significant changes for unconfirmed candles to reduce noise
                        # prev_rsi = self.rsi_values.get(symbol)
                        # if prev_rsi is None or abs(new_rsi - prev_rsi) > 1:
                        #      logger.debug(f"Intermediate RSI for {symbol}: {new_rsi:.2f}")


                # Store and check threshold if RSI is valid
                if not np.isnan(new_rsi):
                     prev_rsi = self.rsi_values.get(symbol, np.nan)
                     self.rsi_values[symbol] = new_rsi
                     if is_closed: # Log final RSI for closed candles
                         logger.info(f"{symbol} RSI updated (CLOSED): {new_rsi:.2f}")
                     # Check threshold if RSI changed significantly or crossed threshold
                     if np.isnan(prev_rsi) or abs(new_rsi - prev_rsi) > 0.1: # Avoid alerts on tiny fluctuations
                           self._check_and_send_alert(symbol, new_rsi)

        except Exception as e:
            logger.error(f"Error processing kline message for {symbol if 'symbol' in locals() else 'unknown symbol'}: {e}", exc_info=True)
            logger.error(f"Problematic message: {message}")


    def _check_and_send_alert(self, symbol, rsi, is_initial=False):
        """Check RSI against thresholds and send notifications if needed."""
        current_time = time.time()
        last_alert = self.last_alert_time.get(symbol, 0)
        alert_type = None
        alert_msg = ""

        if rsi >= OVERBOUGHT_THRESHOLD:
            alert_type = "OVERBOUGHT"
            alert_msg = f"📈 {symbol} RSI is OVERBOUGHT at {rsi:.2f} (Threshold: {OVERBOUGHT_THRESHOLD})"
        elif rsi <= OVERSOLD_THRESHOLD:
            alert_type = "OVERSOLD"
            alert_msg = f"📉 {symbol} RSI is OVERSOLD at {rsi:.2f} (Threshold: {OVERSOLD_THRESHOLD})"

        if alert_type:
            # Send alert if it's the initial check OR if cooldown has passed
            if is_initial or (current_time - last_alert >= ALERT_COOLDOWN_SECONDS):
                logger.warning(f"ALERT Triggered: {alert_msg}")
                self.last_alert_time[symbol] = current_time
                # --- Send Push Notification ---
                self._send_push_notification(title=f"RSI Alert: {symbol}", body=alert_msg)
            # else:
                 # logger.debug(f"Alert condition met for {symbol} ({alert_type} @ {rsi:.2f}), but still in cooldown.")


    def _send_push_notification(self, title: str, body: str):
        """Queries Supabase for tokens and sends notifications via Expo."""
        logger.info(f"Preparing to send push notification: {title} - {body}")
        try:
            # 1. Query Supabase for all push tokens
            # TODO: Add filtering later if user preferences are implemented
            response = self.supabase.table('push_tokens').select('token').execute()

            if not response.data:
                logger.warning("No push tokens found in database. Cannot send notification.")
                return

            tokens = [item['token'] for item in response.data if item.get('token')]
            if not tokens:
                 logger.warning("No valid push tokens extracted from database response.")
                 return

            logger.info(f"Found {len(tokens)} push tokens to notify.")

            # 2. Send notifications to Expo (handle chunking if necessary)
            # Expo recommends sending in chunks of 100
            messages = []
            for token in tokens:
                 # Basic validation: Expo tokens often start with ExponentPushToken[
                 if isinstance(token, str) and token.startswith("ExponentPushToken["):
                     messages.append({
                         "to": token,
                         "sound": "default",
                         "title": title,
                         "body": body,
                         # "data": {"extra": "data"} # Optional extra data
                     })
                 else:
                     logger.warning(f"Skipping invalid token format: {token}")

            if not messages:
                logger.warning("No valid messages to send after filtering tokens.")
                return

            # Send messages in chunks
            chunk_size = 100
            for i in range(0, len(messages), chunk_size):
                 chunk = messages[i:i+chunk_size]
                 try:
                     headers = {
                         'Accept': 'application/json',
                         'Accept-Encoding': 'gzip, deflate',
                         'Content-Type': 'application/json',
                     }
                     response = requests.post(EXPO_PUSH_URL, headers=headers, json=chunk, timeout=10)
                     response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                     # Log success/failure details from Expo response
                     try:
                         result = response.json()
                         # Check result['data'] for status of individual messages if needed
                         logger.info(f"Expo push request successful for chunk {i//chunk_size + 1}. Response status: {response.status_code}")
                         # Add more detailed logging based on Expo's response format if issues arise
                     except requests.exceptions.JSONDecodeError:
                          logger.info(f"Expo push request successful for chunk {i//chunk_size + 1}. Status code: {response.status_code}. Non-JSON response.")

                 except requests.exceptions.RequestException as e:
                      logger.error(f"Error sending push notification chunk {i//chunk_size + 1} to Expo: {e}")
                      # Continue to next chunk if one fails

        except Exception as e:
            # Catch errors from Supabase query or general logic
            logger.error(f"Failed to send push notification: {e}", exc_info=True)


    def _run_monitoring_loop(self):
        """Main loop to manage WebSocket connection and subscriptions."""
        logger.info("Monitoring loop thread started.")
        self.is_running = True
        self._stop_event.clear()

        # Start the heartbeat monitor in its own thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_monitor_thread, daemon=True, name="HeartbeatMon")
        heartbeat_thread.start()

        while not self._stop_event.is_set():
            try:
                # Initialize or reinitialize WebSocket if needed
                if not self.ws or not self.connected:
                     # Calculate backoff before attempting connection/reconnection
                     if self.reconnect_count > 0:
                          backoff = min(2 ** self.reconnect_count, MAX_RECONNECT_DELAY)
                          logger.info(f"Waiting {backoff} seconds before attempting reconnect (attempt {self.reconnect_count + 1})...")
                          time.sleep(backoff)

                     self.init_websocket()
                     if not self.ws:
                         logger.error("Failed to initialize WebSocket. Retrying...")
                         self.reconnect_count += 1
                         continue # Retry after delay

                     # Fetch symbols if needed (usually only first time)
                     if not self.symbols:
                          if not self.initialize_price_history():
                              logger.error("Failed initial price history load. Cannot proceed with subscription.")
                              # Consider a longer retry delay or stopping if this fails repeatedly
                              self.reconnect_count += 1
                              continue

                     # Subscribe to klines
                     if not self.subscribe_to_klines():
                         logger.error("Failed to subscribe to kline streams. Retrying...")
                         self.reconnect_count += 1
                         self.connected = False # Ensure marked as disconnected
                         # Close the potentially partially connected socket before retry
                         if self.ws:
                              self.ws.exit()
                              self.ws = None
                         continue # Retry subscription after delay
                     else:
                          # Success! Reset reconnect count
                          self.reconnect_count = 0
                          self.connected = True


                # If connected, just sleep and let the WS thread handle messages
                # The heartbeat monitor will trigger reconnect if needed
                time.sleep(5) # Check loop status periodically

            except Exception as e:
                 logger.error(f"Exception in monitoring loop: {e}", exc_info=True)
                 self.connected = False # Assume connection lost on error
                 self.reconnect_count += 1
                 # Close WebSocket if it exists to ensure clean state for reconnect attempt
                 if self.ws:
                     try:
                         self.ws.exit()
                     except Exception as ws_exit_e:
                          logger.warning(f"Error closing WebSocket during exception handling: {ws_exit_e}")
                     finally:
                          self.ws = None
                 time.sleep(5) # Wait a bit before retrying the loop


        logger.info("Monitoring loop requested to stop.")
        # Cleanup WebSocket connection
        if self.ws:
            logger.info("Closing WebSocket connection...")
            try:
                self.ws.exit()
            except Exception as e:
                logger.warning(f"Exception closing WebSocket: {e}")
        logger.info("Monitoring loop thread finished.")


    def start_monitoring(self):
        """Starts the monitoring process in a separate thread."""
        if self.is_running:
            logger.warning("Monitoring is already running.")
            return

        # Run the main loop in a background thread
        self.monitor_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True, name="RSIMonitorLoop")
        self.monitor_thread.start()
        logger.info("RSI Monitoring background task started.")


    def stop_monitoring(self):
        """Stops the monitoring process."""
        if not self.is_running:
            logger.warning("Monitoring is not running.")
            return

        logger.info("Stopping RSI monitoring...")
        self.is_running = False
        self._stop_event.set() # Signal threads to stop

        # Wait for threads to finish
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
             logger.info("Waiting for monitoring loop thread to finish...")
             self.monitor_thread.join(timeout=10) # Wait up to 10 seconds
             if self.monitor_thread.is_alive():
                  logger.warning("Monitoring loop thread did not finish gracefully.")
        # Heartbeat thread is daemon, should exit automatically

        logger.info("RSI Monitoring stopped.")


# --- FastAPI Application ---

# Global state for the monitor instance
monitor: RSIMonitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global monitor
    logger.info("FastAPI application startup...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client created.")
        monitor = RSIMonitor(supabase_client=supabase)
        monitor.start_monitoring()
        logger.info("RSI Monitor started in background.")
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        # Depending on severity, might want to prevent app start
        raise HTTPException(status_code=500, detail=f"Startup failed: {e}") from e

    yield # Application runs here

    # Shutdown
    logger.info("FastAPI application shutdown...")
    if monitor:
        monitor.stop_monitoring()
    logger.info("FastAPI application finished shutdown.")


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "RSI Monitor Service running."}

@app.get("/status")
async def get_status():
    """Returns the current status of the RSI monitor."""
    if not monitor:
        return {"status": "error", "message": "Monitor not initialized."}

    with monitor.lock: # Use lock to safely access shared data
        rsi_count = len(monitor.rsi_values)
        tracked_symbols = len(monitor.symbols)
        subscribed_count = len(monitor.subscribed_symbols)

        # Get a snapshot of recent RSI values (e.g., top/bottom 5)
        # Sort items safely, handle potential NaN or missing values
        valid_rsi = {s: r for s, r in monitor.rsi_values.items() if pd.notna(r)}
        sorted_rsi = sorted(valid_rsi.items(), key=lambda item: item[1], reverse=True)
        top_5 = [{"symbol": s, "rsi": r} for s, r in sorted_rsi[:5]]
        bottom_5 = [{"symbol": s, "rsi": r} for s, r in sorted_rsi[-5:]]


    return {
        "status": "running" if monitor.is_running else "stopped",
        "websocket_connected": monitor.connected,
        "last_message_time_ago_s": int(time.time() - monitor.last_message_time) if monitor.last_message_time else None,
        "tracked_symbol_count": tracked_symbols,
        "subscribed_symbol_count": subscribed_count,
        "rsi_calculated_count": rsi_count,
        "reconnect_attempts": monitor.reconnect_count,
        "top_5_rsi": top_5,
        "bottom_5_rsi": bottom_5,
    }

# Add other endpoints if needed, e.g., manually trigger history refresh, etc.


# --- Main Execution Block (for local testing) ---
# This part is not strictly needed when running with Uvicorn via command line
# but can be useful for direct execution/debugging.
if __name__ == "__main__":
    # Note: Running directly like this bypasses the Uvicorn server features
    # It's better to run using: uvicorn main:app --reload (for development)
    logger.warning("Running script directly. For production/proper ASGI, use Uvicorn: `uvicorn main:app --host 0.0.0.0 --port 8000`")

    # Basic setup to run the monitor standalone (without FastAPI server) for debugging
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        standalone_monitor = RSIMonitor(supabase_client)
        standalone_monitor.start_monitoring()

        # Keep the main thread alive
        while True:
            time.sleep(60) # Keep running

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down standalone monitor.")
        if 'standalone_monitor' in locals() and standalone_monitor:
             standalone_monitor.stop_monitoring()
    except Exception as e:
         logger.error(f"Error in standalone execution: {e}", exc_info=True) 