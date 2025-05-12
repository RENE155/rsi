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
WS_PING_INTERVAL = 20  # Send ping every 20 seconds
WS_PING_TIMEOUT = 10   # Wait up to 10 seconds for pong response
HEARTBEAT_INTERVAL = 15 # Check connection every 15s instead of 30s
NO_MESSAGE_TIMEOUT = 60 # Reconnect if no messages for 60s (reduced from 120s)
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
        self.subscribed_symbols = set() # Tracks symbols successfully subscribed via WS
        self.lock = threading.Lock()
        self.is_running = False
        self.rsi_values = defaultdict(dict) # Changed: Stores RSI per interval, e.g., {"BTCUSDT": {"240": 70.5, "D": 65.1}}
        self.last_alert_time = defaultdict(float)
        self.connected = False
        self.reconnect_count = 0
        self.last_message_time = 0
        self.last_heartbeat_check = 0
        self._stop_event = threading.Event()
        self._last_connected_log = 0
        self.funding_rates = {} # Stores current funding rate per symbol
        self.monitor_thread = None # Thread for main WS loop
        self.aux_data_thread = None # Thread for auxiliary data fetching (1D RSI)

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

    def get_rsi_for_interval(self, symbol, interval):
        """
        Calculate RSI for a specific symbol and interval
        
        Args:
            symbol (str): Trading symbol like "BTCUSDT"
            interval (str): Kline interval like "240" (4h) or "D" (1d)
            
        Returns:
            float or None: RSI value or None if error occurs
        """
        try:
            # Removed daily RSI caching logic - now handled by background task/stored in self.rsi_values
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    kline_data = self.session.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        limit=RSI_PERIODS + 50  # Get enough data for RSI calculation
                    )
                    
                    if kline_data.get("retCode", -1) != 0:
                        msg = kline_data.get('retMsg', 'Unknown API error')
                        logger.warning(f"API error getting {interval} kline for {symbol}: {msg}")
                        if "rate limit" in msg.lower():
                            wait_time = 1 * (2 ** attempt)
                            logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        return None
                    
                    if not kline_data.get("result", {}).get("list"):
                        logger.warning(f"No {interval} kline data returned for {symbol}")
                        return None
                        
                    prices = [float(k[4]) for k in kline_data["result"]["list"]]
                    prices.reverse()  # API returns newest first
                    
                    if len(prices) >= RSI_PERIODS + 1:
                        rsi = calculate_rsi(pd.Series(prices))
                        if not np.isnan(rsi):
                            # Don't cache here anymore
                            return rsi
                    else:
                        logger.warning(f"Insufficient data for {symbol} {interval} RSI calculation")
                    
                    return None
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 1 * (2 ** attempt)
                        logger.warning(f"Error getting {interval} RSI for {symbol} (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to get {interval} RSI for {symbol} after {max_retries} attempts: {e}")
                        return None
        except Exception as e:
            logger.error(f"Unexpected error calculating {interval} RSI for {symbol}: {e}", exc_info=True)
            return None

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

            # Define event handlers
            def on_disconnect():
                # This might be called by pybit or our code
                if self.connected:
                    logger.warning("WebSocket disconnected (on_disconnect callback triggered)")
                    self.connected = False
                    self.last_message_time = 0 # Reset timer

            def on_error(error):
                # Log the specific error and mark as disconnected
                logger.error(f"WebSocket error (on_error callback triggered): {error}", exc_info=True)
                self.connected = False
                self.last_message_time = 0 # Reset timer

            def on_close():
                # Log when the connection is closed and mark as disconnected
                if self.connected: # Avoid logging if already marked disconnected
                    logger.info("WebSocket connection closed (on_close callback triggered)")
                    self.connected = False
                    self.last_message_time = 0 # Reset timer

            # Check if pybit WebSocket supports event handlers
            # If not, we'll rely on our heartbeat monitor
            try:
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="linear",
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                    trace_logging=False,
                    on_disconnect=on_disconnect,
                    on_error=on_error,
                    on_close=on_close
                )
                logger.info("WebSocket initialized with event handlers.")
            except TypeError:
                # Handlers not supported, use standard constructor
                logger.warning("WebSocket event handlers (on_error, on_close) not supported by this pybit version. Relying on heartbeat.")
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="linear",
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                    trace_logging=False
                )

            self.connected = False
            self.last_message_time = time.time() # Initialize timer
            logger.info("WebSocket initialization process complete.")
        except Exception as e:
            logger.error(f"WebSocket initialization error: {e}", exc_info=True)
            self.ws = None # Ensure ws is None if init fails


    def _heartbeat_monitor_thread(self):
        """Runs in a thread to monitor WebSocket connection health."""
        logger.info("Heartbeat monitor thread started.")
        last_log_time = 0
        
        while not self._stop_event.is_set():
            current_time = time.time()
            
            # If we have an active connection that hasn't received messages recently
            if (self.ws and self.connected and 
                self.last_message_time > 0 and 
                (current_time - self.last_message_time > NO_MESSAGE_TIMEOUT) and 
                len(self.subscribed_symbols) > 0):
                
                # Log less frequently to avoid flooding logs
                if current_time - last_log_time > 60:  # Log at most once per minute
                    logger.warning(f"Heartbeat Monitor: No messages received for {int(current_time - self.last_message_time)} seconds (threshold: {NO_MESSAGE_TIMEOUT}s). Triggering reconnect.")
                    last_log_time = current_time
                
                # Actively test connection by trying to ping if supported by the library
                try:
                    # Mark connection as disconnected so the main loop will reconnect
                    logger.info("Heartbeat Monitor: Marking connection as disconnected to trigger reconnect.")
                    self.connected = False
                    
                    # Try to close the connection gracefully 
                    if self.ws:
                        try:
                            self.ws.exit()
                        except Exception as e:
                            logger.warning(f"Error closing stale WebSocket: {e}")
                except Exception as e:
                    logger.error(f"Error during connection test: {e}")
            
            # Also do a quick ping test every 30s to ensure connection is responsive
            elif (self.ws and self.connected and 
                  self.last_message_time > 0 and
                  (current_time - self.last_message_time > 30)):
                logger.debug("Performing routine connection check")
                
                # Just mark the time, so we know heartbeat is running
                # The websockets library already handles ping/pong internally
                self.last_heartbeat_check = current_time
                                
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
            self.rsi_values.clear() # Clears the nested dict
            self.funding_rates.clear()

        total_symbols = len(self.symbols)
        logger.info(f"Initializing data for {total_symbols} symbols (4H RSI, 1D RSI, Funding Rate)...")
        initial_fetch_delay = 0.1 # Delay between symbols during initial fetch
        
        for idx, symbol in enumerate(self.symbols, 1):
            if not self.is_running:
                logger.info("Stopping price history initialization.")
                return False

            logger.info(f"Initializing {symbol} ({idx}/{total_symbols})...")
            try:
                # 1. Fetch Historical Kline for Primary Interval (4H)
                kline_data_4h = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        kline_data_4h = self.session.get_kline(
                            category="linear",
                            symbol=symbol,
                            interval=KLINE_INTERVAL, # Use the primary interval (e.g., "240")
                            limit=RSI_PERIODS + 100 # Get enough data for initial RSI + buffer
                        )
                        if kline_data_4h.get("retCode", -1) != 0:
                            msg = kline_data_4h.get('retMsg', 'Unknown API error')
                            logger.warning(f"API error getting {KLINE_INTERVAL} kline for {symbol}: {msg}")
                            if "rate limit" in msg.lower():
                                wait_time = 1 * (2 ** attempt)
                                logger.warning(f"Rate limit hit for {symbol}. Waiting {wait_time}s...")
                                time.sleep(wait_time)
                                continue # Retry
                            else:
                                break # Non-rate-limit error
                        break # Success
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 1 * (2 ** attempt)
                            logger.warning(f"Failed to get {KLINE_INTERVAL} kline for {symbol} (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to get {KLINE_INTERVAL} kline for {symbol} after {max_retries} attempts.")
                            # Continue to next step even if this fails, maybe log symbol as problematic
                            break 
                
                # Process 4H Kline Data
                if kline_data_4h and kline_data_4h.get("retCode") == 0 and kline_data_4h.get("result", {}).get("list"):
                    prices = [float(k[4]) for k in kline_data_4h["result"]["list"]]
                    prices.reverse() # API returns newest first

                    if len(prices) >= RSI_PERIODS + 1:
                        with self.lock:
                            self.price_data[symbol] = prices # Store price history for WS updates
                            rsi_4h = calculate_rsi(pd.Series(prices))
                            if not np.isnan(rsi_4h):
                                self.rsi_values[symbol][KLINE_INTERVAL] = round(rsi_4h, 2)
                                logger.debug(f"Initial {KLINE_INTERVAL} RSI for {symbol}: {self.rsi_values[symbol][KLINE_INTERVAL]:.2f}")
                                # Check initial threshold (based on primary interval)
                                self._check_and_send_alert(symbol, rsi_4h, is_initial=True)
                    else:
                        logger.warning(f"Insufficient historical data for {symbol} {KLINE_INTERVAL} RSI ({len(prices)} points). Need {RSI_PERIODS + 1}.")
                else:
                     logger.warning(f"Could not retrieve valid {KLINE_INTERVAL} kline data for {symbol}. API Response: {kline_data_4h}")

                # 2. Fetch Initial 1D RSI (using the modified function)
                try:
                    rsi_1d = self.get_rsi_for_interval(symbol, "D")
                    if rsi_1d is not None and not np.isnan(rsi_1d):
                         with self.lock:
                             self.rsi_values[symbol]["D"] = round(rsi_1d, 2)
                         logger.debug(f"Initial 1D RSI for {symbol}: {self.rsi_values[symbol]['D']:.2f}")
                    # else: logger.warning(f"Could not get initial 1D RSI for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting initial 1D RSI for {symbol}: {e}")

                # 3. Fetch Initial Funding Rate (one REST poll at startup)
                # WebSocket tickers stream will take over after connection
                try:
                    funding_rate = self._get_current_funding_rate(symbol)
                    if funding_rate is not None:
                        with self.lock:
                             self.funding_rates[symbol] = funding_rate
                        logger.debug(f"Initial Funding Rate for {symbol}: {funding_rate}")
                    # else: logger.warning(f"Could not get initial funding rate for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting initial funding rate for {symbol}: {e}")

                # Delay before processing next symbol
                time.sleep(initial_fetch_delay)

            except Exception as e:
                logger.error(f"Error initializing data for {symbol}: {e}", exc_info=True)
                continue # Continue with the next symbol

        logger.info("Initial data fetching complete.")
        return True # Indicate success

    def subscribe_to_streams(self):
        """Subscribe to WebSocket kline and ticker streams for all symbols"""
        if not self.ws:
            logger.error("WebSocket not initialized. Cannot subscribe.")
            return False

        if not self.symbols:
            logger.warning("Symbols list is empty. Attempting to fetch.")
            self.symbols = self._get_all_symbols()
            if not self.symbols:
                logger.error("Cannot subscribe: Failed to fetch symbols.")
                return False

        logger.info(f"Starting WebSocket subscriptions for {len(self.symbols)} symbols (Klines: {KLINE_INTERVAL}, Tickers)...")

        def handle_kline_message_wrapper(message):
            """Wrapper to update last message time and call kline handler"""
            self.last_message_time = time.time()
            if not hasattr(self, '_last_connected_log') or time.time() - self._last_connected_log > 60:
                if not self.connected:
                    logger.info("WS receiving kline messages - marking as connected")
                self._last_connected_log = time.time()
            self.connected = True
            self._handle_kline_message(message)
        
        def handle_ticker_message_wrapper(message):
            """Wrapper to update last message time and call ticker handler"""
            # Ticker messages update last_message_time and connected status inside _handle_ticker_message
            self._handle_ticker_message(message)

        # Clear previous subscriptions before starting new ones
        self.subscribed_symbols.clear()
        
        # Subscribe in batches 
        batch_size = 20  # Reduced batch size further for safety with two stream types
        successfully_subscribed = set()
        total_batches = (len(self.symbols) + batch_size - 1) // batch_size
        
        for i in range(0, len(self.symbols), batch_size):
            batch_symbols = self.symbols[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Attempting subscriptions for batch {batch_num}/{total_batches} ({len(batch_symbols)} symbols)..." )
            
            subscriptions_in_batch = 0
            try:
                # Subscribe to Klines for the batch
                kline_args = [f"kline.{KLINE_INTERVAL}.{s}" for s in batch_symbols]
                self.ws.subscribe(kline_args, callback=handle_kline_message_wrapper)
                logger.debug(f"Batch {batch_num}: Subscribed to {len(kline_args)} kline streams.")
                subscriptions_in_batch += len(kline_args)
                time.sleep(0.5) # Small delay between stream types within batch

                # Subscribe to Tickers for the batch
                ticker_args = [f"tickers.{s}" for s in batch_symbols]
                self.ws.subscribe(ticker_args, callback=handle_ticker_message_wrapper)
                logger.debug(f"Batch {batch_num}: Subscribed to {len(ticker_args)} ticker streams.")
                subscriptions_in_batch += len(ticker_args)
                
                # Assume success for the symbols in this batch if no exception
                successfully_subscribed.update(batch_symbols)
                logger.info(f"Batch {batch_num}: Submitted {subscriptions_in_batch} subscriptions. Total symbols subscribed: {len(successfully_subscribed)}")

            except Exception as e:
                logger.error(f"Error subscribing streams in batch {batch_num}: {e}", exc_info=True)
                # Consider which symbols failed if possible, but for now, log and continue
            
            # Wait before next batch
            if i + batch_size < len(self.symbols):
                 logger.info("Waiting 2 seconds before next batch...")
                 time.sleep(2)

        self.subscribed_symbols = successfully_subscribed
        if not self.subscribed_symbols:
             logger.error("Failed to subscribe to any streams after processing all batches.")
             return False

        logger.info(f"Successfully submitted subscriptions for {len(self.subscribed_symbols)} symbols.")
        self.last_message_time = time.time() # Reset timer after successful subscriptions
        # self.connected will be set to True by the first message received by wrappers
        self.reconnect_count = 0 # Reset reconnect counter on successful subscription attempt
        return True

    def _handle_kline_message(self, message):
        """Process incoming kline WebSocket messages and update RSI values for KLINE_INTERVAL"""
        try:
            topic = message.get("topic", "")
            data_list = message.get("data", [])

            if not topic.startswith(f"kline.{KLINE_INTERVAL}.") or not data_list:
                # logger.debug(f"Ignoring non-kline or empty message for interval {KLINE_INTERVAL}: {message}")
                return

            # Extract symbol from topic, e.g., "kline.240.BTCUSDT"
            try:
                symbol = topic.split(".")[-1]
            except IndexError:
                logger.warning(f"Could not extract symbol from kline topic: {topic}")
                return

            if symbol not in self.symbols:
                 logger.warning(f"Received kline message for unexpected/untracked symbol: {symbol}")
                 return # Ignore symbols we didn't intend to track

            data = data_list[0] # Data is usually a list containing one dictionary
            is_closed = data.get("confirm", False)
            current_price = float(data["close"])
            # timestamp_ms = int(data["start"]) # Start time of the candle - potentially useful later

            with self.lock:
                current_prices = self.price_data.get(symbol, [])
                new_rsi = np.nan

                if is_closed:
                     # Add the closing price
                     self.price_data[symbol].append(current_price)
                     # Keep buffer size reasonable
                     if len(self.price_data[symbol]) > RSI_PERIODS + 200:
                         self.price_data[symbol] = self.price_data[symbol][-(RSI_PERIODS + 200):]
                     # logger.debug(f"CLOSED candle for {symbol} ({KLINE_INTERVAL}). Price: {current_price}. History length: {len(self.price_data[symbol])}")
                     # Recalculate RSI on closed candle
                     price_series = pd.Series(self.price_data[symbol])
                     new_rsi = calculate_rsi(price_series)
                else: # Unconfirmed (intermediate) candle update
                    if not current_prices:
                        # Should ideally not happen if history is initialized, but handle defensively
                        self.price_data[symbol].append(current_price)
                        # logger.debug(f"First price point (unconfirmed) for {symbol} ({KLINE_INTERVAL}): {current_price}")
                    else:
                        # Create a temporary series with the latest price updated
                        temp_prices = current_prices[:-1] + [current_price]
                        price_series = pd.Series(temp_prices)
                        new_rsi = calculate_rsi(price_series)

                # Store and check threshold if RSI is valid
                if not np.isnan(new_rsi):
                     new_rsi_rounded = round(new_rsi, 2)
                     prev_rsi = self.rsi_values.get(symbol, {}).get(KLINE_INTERVAL)
                     # Update the specific interval in the nested dict
                     self.rsi_values[symbol][KLINE_INTERVAL] = new_rsi_rounded
                     
                     if is_closed: # Log final RSI for closed candles
                         logger.info(f"{symbol} RSI ({KLINE_INTERVAL}) updated (CLOSED): {new_rsi_rounded:.2f}")
                     
                     # Check threshold ONLY for the primary interval (KLINE_INTERVAL)
                     if ( prev_rsi is None
                          or abs(new_rsi_rounded - prev_rsi) > 0.1 # Check if changed significantly
                          or (prev_rsi < OVERBOUGHT_THRESHOLD and new_rsi_rounded >= OVERBOUGHT_THRESHOLD)
                          or (prev_rsi > OVERSOLD_THRESHOLD and new_rsi_rounded <= OVERSOLD_THRESHOLD)
                      ):
                         self._check_and_send_alert(symbol, new_rsi_rounded)

        except Exception as e:
            logger.error(f"Error processing kline message for {symbol if 'symbol' in locals() else 'unknown symbol'}: {e}", exc_info=True)
            logger.error(f"Problematic kline message: {message}")


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

            tokens_raw = [item['token'] for item in response.data if item.get('token')]
            if not tokens_raw:
                 logger.warning("No valid push tokens extracted from database response.")
                 return

            # Convert to a set to get unique tokens, then back to a list
            unique_tokens = list(set(tokens_raw))

            logger.info(f"Found {len(unique_tokens)} unique push tokens to notify (out of {len(tokens_raw)} total).")

            # 2. Send notifications to Expo (handle chunking if necessary)
            # Expo recommends sending in chunks of 100
            messages = []
            # Iterate over unique_tokens instead of tokens
            for token in unique_tokens:
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


    def _handle_ticker_message(self, message):
        """Process incoming ticker WebSocket messages and update funding rates."""
        try:
            if message.get("topic", "").startswith("tickers.") and message.get("type") == "snapshot":
                data = message.get("data", {})
                symbol = data.get("symbol")
                funding_rate_str = data.get("fundingRate")

                if symbol and funding_rate_str and symbol in self.symbols:
                    try:
                        funding_rate = float(funding_rate_str)
                        with self.lock:
                            if self.funding_rates.get(symbol) != funding_rate:
                                # logger.debug(f"Funding rate for {symbol} updated via WS: {funding_rate}")
                                self.funding_rates[symbol] = funding_rate
                        self.last_message_time = time.time() # Also update last message time on ticker updates
                        self.connected = True
                    except ValueError:
                        logger.warning(f"Could not convert funding rate '{funding_rate_str}' from ticker WS for {symbol}")
            # else:
                # logger.debug(f"Ignoring non-ticker snapshot message: {message}")

        except Exception as e:
            logger.error(f"Error processing ticker message: {e}", exc_info=True)
            logger.error(f"Problematic ticker message: {message}")

    def _get_current_funding_rate(self, symbol: str) -> float | None:
        """
        Fetches the current funding rate for a single symbol using the shared session.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").

        Returns:
            The current funding rate as a float, or None if an error occurs or not found.
        """
        # Note: No need to check API keys here as self.session requires them at init
        try:
            # Use a short timeout for status checks to avoid blocking
            result = self.session.get_tickers(category="linear", symbol=symbol)

            if result and result.get("retCode") == 0:
                ticker_info = result.get("result", {}).get("list", [])
                if ticker_info:
                    funding_rate_str = ticker_info[0].get("fundingRate")
                    if funding_rate_str and funding_rate_str != "": # Ensure rate exists and is not empty string
                        try:
                            return float(funding_rate_str)
                        except ValueError:
                            logger.warning(f"Could not convert funding rate '{funding_rate_str}' to float for {symbol}.")
                            return None
                    else:
                        # logger.debug(f"Funding rate key missing or empty in ticker info for {symbol}.")
                        return None # Explicitly return None if key missing or empty
                else:
                    # logger.debug(f"No ticker list returned in API response for {symbol}.")
                    return None
            else:
                # Log API errors, but maybe less verbosely for status checks unless debugging
                # Avoid flooding logs if the endpoint is hit frequently
                if result.get("retCode") != 10006: # 10006 is often a temporary timeout/overload error
                     logger.warning(f"Bybit API error fetching ticker for {symbol} funding rate: {result.get('retMsg', 'Unknown error')} (Code: {result.get('retCode')})")
                return None

        except Exception as e:
            # Log other exceptions like connection errors
            logger.error(f"Exception fetching funding rate for {symbol}: {e}", exc_info=False) # Limit traceback spam
            return None

    def _update_auxiliary_data_loop(self):
        """Runs in a thread to periodically fetch 1D RSI for all symbols."""
        logger.info("Auxiliary data fetching loop started (1D RSI). Interval: 300s")
        fetch_interval_seconds = 300 # Fetch every 5 minutes
        api_call_delay = 0.2 # Delay between API calls within the loop

        while not self._stop_event.is_set():
            start_time = time.time()
            logger.info("Starting periodic 1D RSI update cycle...")
            symbols_to_update = self.symbols # Get current list of symbols
            updated_count = 0
            error_count = 0

            for symbol in symbols_to_update:
                if self._stop_event.is_set():
                    break # Exit loop if stop signal received

                try:
                    # Use get_rsi_for_interval (which no longer caches internally)
                    daily_rsi = self.get_rsi_for_interval(symbol, "D")
                    
                    if daily_rsi is not None and not np.isnan(daily_rsi):
                        with self.lock: # Acquire lock to update shared dict
                            self.rsi_values[symbol]["D"] = round(daily_rsi, 2)
                        updated_count += 1
                        # logger.debug(f"Updated 1D RSI for {symbol}: {self.rsi_values[symbol]['D']}")
                    else:
                        # Optionally handle cases where RSI couldn't be calculated (e.g., log, keep old value)
                        # logger.warning(f"Could not get 1D RSI for {symbol}")
                        pass 
                except Exception as e:
                    logger.error(f"Error fetching 1D RSI for {symbol} in background task: {e}")
                    error_count += 1
                
                # Wait before the next symbol to avoid rate limits
                time.sleep(api_call_delay)

            cycle_duration = time.time() - start_time
            logger.info(f"Periodic 1D RSI update cycle finished in {cycle_duration:.2f}s. Updated: {updated_count}, Errors: {error_count}")

            # Wait for the next interval, accounting for cycle duration
            wait_time = max(0, fetch_interval_seconds - cycle_duration)
            if wait_time > 0:
                # Use event wait for faster shutdown response
                self._stop_event.wait(timeout=wait_time)

        logger.info("Auxiliary data fetching loop stopped.")

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

                     # Subscribe to streams
                     if not self.subscribe_to_streams():
                         logger.error("Failed to subscribe to streams. Retrying...")
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
                # Check connection status before sleeping
                if not self.connected:
                    logger.info("Main Loop: Detected disconnected state. Initiating reconnection sequence.")
                    # No need to sleep, the loop will restart the connection attempt
                    continue
                    
                time.sleep(5) # Check loop status periodically

            except (ConnectionRefusedError, ConnectionResetError) as conn_e:
                # Specific network connection errors
                logger.error(f"Main Loop: Network connection error ({type(conn_e).__name__}): {conn_e}")
                self.connected = False
                self.reconnect_count += 1
                if self.ws:
                    try: self.ws.exit() 
                    except Exception: pass # Ignore errors during exit
                    finally: self.ws = None
                # Backoff handled at the start of the loop
                
            except Exception as e:
                 # General catch-all for other unexpected errors
                 logger.error(f"Main Loop: General exception ({type(e).__name__}): {e}", exc_info=True)
                 self.connected = False # Assume connection lost on error
                 self.reconnect_count += 1


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
        
        logger.info("Starting RSI Monitoring background tasks...")
        self.is_running = True # Set flag early
        self._stop_event.clear()

        # Start the main WebSocket monitoring loop thread
        self.monitor_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True, name="RSIMonitorLoop")
        self.monitor_thread.start()
        
        # Start the auxiliary data fetching loop thread (for 1D RSI etc.)
        self.aux_data_thread = threading.Thread(target=self._update_auxiliary_data_loop, daemon=True, name="AuxDataLoop")
        self.aux_data_thread.start()

        logger.info("RSI Monitoring background tasks started.")


    def stop_monitoring(self):
        """Stops the monitoring process."""
        if not self.is_running:
            logger.warning("Monitoring is not running.")
            return

        logger.info("Stopping RSI monitoring...")
        self._stop_event.set() # Signal ALL threads using this event to stop
        self.is_running = False # Set flag

        # Wait for threads to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
             logger.info("Waiting for WebSocket monitoring loop thread to finish...")
             self.monitor_thread.join(timeout=10) # Wait up to 10 seconds
             if self.monitor_thread.is_alive():
                  logger.warning("WebSocket monitoring loop thread did not finish gracefully.")
        
        if self.aux_data_thread and self.aux_data_thread.is_alive():
             logger.info("Waiting for auxiliary data fetching loop thread to finish...")
             self.aux_data_thread.join(timeout=10) # Wait up to 10 seconds
             if self.aux_data_thread.is_alive():
                  logger.warning("Auxiliary data fetching loop thread did not finish gracefully.")
        
        # Heartbeat thread is daemon, should exit automatically if its loop checks _stop_event
        # Ensure WebSocket is closed if monitor loop didn't handle it
        if self.ws:
            logger.info("Attempting final WebSocket closure...")
            try: self.ws.exit()
            except Exception as e: logger.warning(f"Exception during final WebSocket closure: {e}")

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
async def get_status(
    interval: str = KLINE_INTERVAL, # Default to the primary WS interval (e.g., "240")
    sort_by: str = "rsi", # "rsi" or "funding_rate"
    limit: int = 5
):
    """Returns the current status, top/bottom symbols by RSI or Funding Rate."""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized.")
    
    # Validate parameters
    valid_intervals = {KLINE_INTERVAL, "D", "60"} # Add "60" if 1H RSI is fetched by background task
    if interval not in valid_intervals:
         raise HTTPException(status_code=400, detail=f"Invalid interval. Valid options: {valid_intervals}")
    if sort_by not in ["rsi", "funding_rate"]:
        raise HTTPException(status_code=400, detail="Invalid sort_by. Valid options: 'rsi', 'funding_rate'")
    if not (1 <= limit <= 20): # Set a reasonable limit range
        raise HTTPException(status_code=400, detail="Invalid limit. Must be between 1 and 20.")

    all_data = []
    with monitor.lock:  # Use lock to safely access shared data
        # Create a snapshot of symbols to iterate over
        current_symbols = list(monitor.symbols)
        
        # Gather data from cache
        for symbol in current_symbols:
            symbol_data = monitor.rsi_values.get(symbol, {})
            funding_rate = monitor.funding_rates.get(symbol)
            
            rsi_requested = symbol_data.get(interval) # RSI for the requested interval (e.g., 240 or D)
            rsi_1d = symbol_data.get("D") # Always include 1D RSI
            
            # Determine primary value for sorting
            primary_value = None
            if sort_by == "rsi":
                primary_value = rsi_requested
            elif sort_by == "funding_rate":
                primary_value = funding_rate
            
            # Only include symbols that have the primary value needed for sorting
            if primary_value is not None and pd.notna(primary_value):
                all_data.append({
                    "symbol": symbol,
                    "rsi_requested": rsi_requested, # The RSI value matching the interval param
                    "rsi_1d": rsi_1d,
                    "funding_rate": funding_rate,
                    "primary_value": primary_value # Value used for sorting
                })
    
    # Define sort key function
    # Sort Nones last (ascending) or first (descending)
    sort_reverse = True # Default: Higher RSI/Funding Rate is "top"
    if sort_by == "rsi" and interval == "D":
        # Optional: For Daily RSI, maybe lower is considered "better" for oversold
        # sort_reverse = False # Uncomment if lower daily RSI should be top
        pass
    elif sort_by == "funding_rate":
        # Higher funding rate (positive) is usually top for payers
        # Lower funding rate (negative) is bottom for receivers
        pass
        
    # Sort handling Nones: place them at the less desirable end
    all_data.sort(
        key=lambda x: (x['primary_value'] is None, x['primary_value']),
        reverse=sort_reverse
    )

    # Select top and bottom N
    top_n_raw = all_data[:limit]
    bottom_n_raw = all_data[-limit:]

    # Format results
    def format_result(item):
        rate = item['funding_rate']
        # Use rsi_requested for the main "rsi" field in the response
        rsi_value = item['rsi_requested'] 
        # Always include the 1D RSI as 'daily_rsi'
        daily_rsi_value = item['rsi_1d']
        return {
            "symbol": item['symbol'],
            "rsi": round(rsi_value, 2) if rsi_value is not None and pd.notna(rsi_value) else None,
            "daily_rsi": round(daily_rsi_value, 2) if daily_rsi_value is not None and pd.notna(daily_rsi_value) else None,
            "funding_rate": rate,
            "funding_rate_percent": f"{rate * 100:.4f}%" if rate is not None and pd.notna(rate) else None
        }

    results_top_n = [format_result(item) for item in top_n_raw]
    results_bottom_n = [format_result(item) for item in bottom_n_raw]
    
    # --- Status Information --- (Also read under lock if possible, or accept minor race condition)
    time_since_last_msg = int(time.time() - monitor.last_message_time) if monitor.last_message_time else None
    websocket_status = "disconnected"
    if monitor.connected:
        if time_since_last_msg is not None and time_since_last_msg <= NO_MESSAGE_TIMEOUT:
             websocket_status = "connected"
        else:
             websocket_status = "stale" # Connected but no recent messages
             
    rsi_calculated_count = len(monitor.rsi_values) # Number of symbols with any RSI data
    funding_rate_count = len(monitor.funding_rates)

    return {
        "status": "running" if monitor.is_running else "stopped",
        "websocket_status": websocket_status,
        "websocket_connected": monitor.connected,
        "last_message_time_ago_s": time_since_last_msg,
        "tracked_symbol_count": len(monitor.symbols),
        "subscribed_symbol_count": len(monitor.subscribed_symbols),
        "rsi_data_count": rsi_calculated_count, # Renamed for clarity
        "funding_rate_data_count": funding_rate_count,
        "reconnect_attempts": monitor.reconnect_count,
        "params": {"sort_by": sort_by, "interval": interval, "limit": limit}, # Reflect params used
        f"top_{limit}_{sort_by}": results_top_n,
        f"bottom_{limit}_{sort_by}": results_bottom_n,
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