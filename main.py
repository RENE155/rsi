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
        self.subscribed_symbols = set() # Tracks symbols with successful kline subscriptions
        self.subscribed_tickers = set() # Tracks symbols with successful ticker subscriptions
        self.lock = threading.Lock() # Lock for accessing shared data (price_data, rsi_values, funding_rates)
        self.is_running = False
        self.rsi_values = {} # Stores latest 4H RSI
        self.funding_rates = {} # Stores latest funding rate from WS
        self.last_alert_time = defaultdict(float)
        self.connected = False
        self.reconnect_count = 0
        self.last_message_time = 0
        self.last_heartbeat_check = 0
        self._stop_event = threading.Event()
        self._last_connected_log = 0
        self.daily_rsi_cache = {} # Cache for daily RSI values
        self.daily_rsi_cache_time = {} # Cache timestamps for daily RSI values

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
            # Check cache for daily values to reduce API calls
            if interval == "D":
                cache_key = f"{symbol}_{interval}"
                # Return cached value if less than 1 hour old
                now = time.time()
                if (cache_key in self.daily_rsi_cache and 
                    cache_key in self.daily_rsi_cache_time and
                    now - self.daily_rsi_cache_time[cache_key] < 3600):
                    logger.debug(f"Using cached daily RSI for {symbol}: {self.daily_rsi_cache[cache_key]}")
                    return self.daily_rsi_cache[cache_key]
            
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
                            # Cache daily RSI values
                            if interval == "D":
                                self.daily_rsi_cache[f"{symbol}_{interval}"] = rsi
                                self.daily_rsi_cache_time[f"{symbol}_{interval}"] = time.time()
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
            self.rsi_values.clear()
            # Initialize funding rates dictionary (optional, could be populated by first WS message)
            # self.funding_rates.clear()

        total_symbols = len(self.symbols)
        processed_count = 0
        fetch_start_time = time.time()
        
        # Fetch initial funding rates via REST API for faster startup
        # This populates the funding_rates dict before WS messages might arrive
        initial_funding_rates = {}
        try:
            logger.info("Fetching initial funding rates for all symbols...")
            tickers_data = self.session.get_tickers(category="linear")
            if tickers_data.get("retCode") == 0 and tickers_data.get("result", {}).get("list"):
                for ticker in tickers_data["result"]["list"]:
                    symbol = ticker.get("symbol")
                    funding_rate_str = ticker.get("fundingRate")
                    if symbol in self.symbols and funding_rate_str and funding_rate_str != "":
                        try:
                            initial_funding_rates[symbol] = float(funding_rate_str)
                        except ValueError:
                            pass # Ignore conversion errors
                logger.info(f"Fetched initial funding rates for {len(initial_funding_rates)} symbols.")
                with self.lock:
                    self.funding_rates.update(initial_funding_rates)
            else:
                logger.warning(f"Could not retrieve initial tickers data. Relying on WebSocket. API Response: {tickers_data}")
        except Exception as e:
            logger.error(f"Error fetching initial tickers data: {e}", exc_info=True)


        logger.info("Initializing 4H RSI history...")
        for idx, symbol in enumerate(self.symbols, 1):
            if not self.is_running:
                logger.info("Stopping price history initialization.")
                return False

            # logger.info(f"Initializing {symbol} ({idx}/{total_symbols})...") # Reduce log verbosity
            try:
                max_retries = 3
                kline_data = None # Initialize kline_data
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
                            if "rate limit" in msg.lower() or kline_data.get("retCode") == 10006:
                                wait_time = 1 * (2 ** attempt)
                                # logger.warning(f"Rate limit/timeout hit for {symbol}. Waiting {wait_time}s...")
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
                            # logger.error(f"Failed to get kline for {symbol} after {max_retries} attempts.", exc_info=True)
                            raise # Reraise after max retries

                if kline_data and kline_data.get("retCode") == 0 and kline_data.get("result", {}).get("list"):
                    prices = [float(k[4]) for k in kline_data["result"]["list"]]
                    prices.reverse() # API returns newest first

                    if len(prices) >= RSI_PERIODS + 1:
                        with self.lock:
                            self.price_data[symbol] = prices
                            rsi = calculate_rsi(pd.Series(prices))
                            if not np.isnan(rsi):
                                self.rsi_values[symbol] = rsi
                                # logger.info(f"Initial RSI for {symbol}: {rsi:.2f}") # Reduce verbosity
                                # Check initial threshold without cooldown
                                self._check_and_send_alert(symbol, rsi, is_initial=True)
                        # logger.info(f"Loaded {len(prices)} historical prices for {symbol}. Initial RSI: {self.rsi_values.get(symbol, 'N/A')}")
                    # else:
                        # logger.warning(f"Insufficient historical data for {symbol} ({len(prices)} points). Need {RSI_PERIODS + 1}.")
                # else:
                     # logger.warning(f"Could not retrieve valid kline data for {symbol}. API Response: {kline_data}")

                processed_count += 1
                # Adaptive delay
                base_delay = 0.05
                progress_factor = 1 - (processed_count / total_symbols)
                delay = base_delay + (progress_factor * 0.1)
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Error initializing price history for {symbol}: {e}") # Reduce log spam
                continue # Continue with the next symbol
        
        init_duration = time.time() - fetch_start_time
        logger.info(f"Historical data initialization complete for {processed_count}/{total_symbols} symbols in {init_duration:.2f} seconds.")
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

        logger.info("Starting WebSocket kline and ticker subscriptions...")

        # --- Define unified message handler --- 
        def handle_message_wrapper(message):
            """Wrapper to update last message time and route to specific handlers"""
            # Update connection state and last message time
            self.last_message_time = time.time()
            
            # Only log this once per minute to avoid log spam
            if not hasattr(self, '_last_connected_log') or time.time() - self._last_connected_log > 60:
                if not self.connected:
                    logger.info("WebSocket is receiving messages - marking as connected")
                self._last_connected_log = time.time()
            
            self.connected = True  # Mark as connected once messages flow
            
            # --- Route message based on topic --- 
            topic = message.get("topic", "")
            if topic.startswith("kline."):
                self._handle_kline_message(message)
            elif topic.startswith("tickers."):
                self._handle_ticker_message(message)
            # else: 
                # logger.debug(f"Ignoring message with unknown topic: {topic}")

        # --- Clear previous subscriptions --- 
        # Note: pybit might handle unsubscribing internally when resubscribing, 
        # but clearing our tracking sets is good practice.
        self.subscribed_symbols.clear()
        self.subscribed_tickers.clear()
        
        # --- Subscribe in batches --- 
        batch_size = 25 
        successfully_subscribed_kline = set()
        successfully_subscribed_ticker = set()
        total_batches = (len(self.symbols) + batch_size - 1) // batch_size

        for i in range(0, len(self.symbols), batch_size):
            batch_symbols = self.symbols[i:i+batch_size] 
            batch_num = i // batch_size + 1
            logger.info(f"Attempting to subscribe symbols in batch {batch_num}/{total_batches}...")
            
            kline_subscribed_in_batch = 0
            ticker_subscribed_in_batch = 0
            
            for symbol in batch_symbols:
                # Subscribe to Kline Stream
                try:
                    self.ws.kline_stream(
                        symbol=symbol,
                        interval=KLINE_INTERVAL,
                        callback=handle_message_wrapper
                    )
                    successfully_subscribed_kline.add(symbol)
                    kline_subscribed_in_batch += 1
                except Exception as e:
                    logger.error(f"Error subscribing to kline for {symbol} in batch {batch_num}: {e}")
                
                # Subscribe to Ticker Stream (for funding rate)
                try:
                    self.ws.ticker_stream(
                        symbol=symbol,
                        callback=handle_message_wrapper
                    )
                    successfully_subscribed_ticker.add(symbol)
                    ticker_subscribed_in_batch += 1
                except Exception as e:
                    logger.error(f"Error subscribing to ticker for {symbol} in batch {batch_num}: {e}")
                
                # Small delay between individual symbol subscriptions in a batch
                time.sleep(0.05)

            logger.info(f"Batch {batch_num}: Subscribed {kline_subscribed_in_batch} klines, {ticker_subscribed_in_batch} tickers.")
            
            # Add a delay between sending batches of subscription requests
            if i + batch_size < len(self.symbols):
                 logger.info(f"Waiting 2 seconds before next batch ({batch_num+1}/{total_batches})...")
                 time.sleep(2)

        # --- Finalize subscription status --- 
        self.subscribed_symbols = successfully_subscribed_kline
        self.subscribed_tickers = successfully_subscribed_ticker
        
        if not self.subscribed_symbols and not self.subscribed_tickers:
             logger.error("Failed to subscribe to ANY streams after processing all batches.")
             return False
        elif not self.subscribed_symbols:
            logger.warning("Failed to subscribe to any KLINE streams.")
        elif not self.subscribed_tickers:
             logger.warning("Failed to subscribe to any TICKER streams.")

        total_subscribed = len(self.subscribed_symbols | self.subscribed_tickers)
        logger.info(f"Successfully subscribed to {len(self.subscribed_symbols)} kline and {len(self.subscribed_tickers)} ticker streams ({total_subscribed} unique symbols).")
        
        self.last_message_time = time.time() # Reset timer after successful subscriptions
        self.connected = True
        self.reconnect_count = 0 # Reset reconnect counter on successful subscription
        return True

    def _handle_kline_message(self, message):
        """Process incoming kline WebSocket messages and update RSI values"""
        try:
            # Basic validation
            if "topic" not in message or "data" not in message or not message["data"]:
                return

            topic = message["topic"]
            # Expected format: kline.{interval}.{symbol} (e.g., kline.240.BTCUSDT)
            parts = topic.split('.')
            if len(parts) < 3:
                 logger.warning(f"Ignoring kline message with unexpected topic format: {topic}")
                 return
            symbol = parts[2] 
            interval = parts[1]
            
            # Ignore if not the interval we are tracking or symbol not monitored
            if interval != KLINE_INTERVAL or symbol not in self.symbols:
                 return

            # Assuming data is a list, take the first element
            data = message["data"][0]
            is_closed = data.get("confirm", False)
            current_price = float(data["close"])
            timestamp_ms = int(data["start"]) # Start time of the candle

            with self.lock:
                current_prices = self.price_data.get(symbol, [])
                new_rsi = np.nan

                if is_closed:
                    # Logic for closed candle update (add price, recalc RSI)
                    # Check timestamp to avoid duplicates if needed
                    # Ensure we don't add old candles if messages arrive out of order
                    last_timestamp = getattr(self.price_data.get(symbol, [{}])[-1], 'timestamp_ms', 0)
                    if not current_prices or timestamp_ms > last_timestamp:
                        # Store price with timestamp for potential future duplicate check
                        # For simplicity now, just append the price float
                        self.price_data[symbol].append(current_price)
                        # Keep buffer size reasonable
                        if len(self.price_data[symbol]) > RSI_PERIODS + 200:
                            self.price_data[symbol] = self.price_data[symbol][-(RSI_PERIODS + 200):]
                        logger.debug(f"CLOSED candle for {symbol}. Price: {current_price}. History length: {len(self.price_data[symbol])}")
                        # Recalculate RSI on closed candle
                        price_series = pd.Series(self.price_data[symbol])
                        new_rsi = calculate_rsi(price_series)
                    else:
                         logger.debug(f"Ignoring older/duplicate closed candle for {symbol} at {timestamp_ms}")
                else: # Unconfirmed (intermediate) candle update
                    # Logic for intermediate update (update last price, recalc RSI)
                    if not current_prices:
                        # Should ideally not happen if history is initialized, but handle defensively
                        self.price_data[symbol].append(current_price)
                        logger.debug(f"First price point (unconfirmed) for {symbol}: {current_price}")
                        price_series = pd.Series(self.price_data[symbol])
                    else:
                        # Create a temporary series with the latest price updated
                        temp_prices = current_prices[:-1] + [current_price]
                        price_series = pd.Series(temp_prices)
                    
                    new_rsi = calculate_rsi(price_series)
                    # Optional: Log only significant changes for unconfirmed candles to reduce noise
                    # prev_rsi = self.rsi_values.get(symbol)
                    # if not np.isnan(new_rsi) and (np.isnan(prev_rsi) or abs(new_rsi - prev_rsi) > 1):
                    #      logger.debug(f"Intermediate RSI for {symbol}: {new_rsi:.2f}")

                # Store and check threshold if RSI is valid
                if not np.isnan(new_rsi):
                     prev_rsi = self.rsi_values.get(symbol, np.nan)
                     self.rsi_values[symbol] = new_rsi
                     # Log final RSI for closed candles more prominently
                     if is_closed:
                         logger.info(f"{symbol} RSI updated (CLOSED): {new_rsi:.2f}")
                     # Check threshold if RSI changed significantly or crossed threshold
                     if (
                         np.isnan(prev_rsi)
                         or abs(new_rsi - prev_rsi) > 0.1 
                         or (prev_rsi < OVERBOUGHT_THRESHOLD and new_rsi >= OVERBOUGHT_THRESHOLD)
                         or (prev_rsi > OVERSOLD_THRESHOLD and new_rsi <= OVERSOLD_THRESHOLD)
                     ):
                         self._check_and_send_alert(symbol, new_rsi)

        except Exception as e:
            # Safely get symbol for logging
            symbol_for_log = "unknown_symbol"
            try:
                topic = message.get("topic", "")
                parts = topic.split('.')
                if len(parts) >= 3:
                    symbol_for_log = parts[2]
            except Exception:
                pass 
            logger.error(f"Error processing kline message for {symbol_for_log}: {e}", exc_info=True)
            # logger.error(f"Problematic kline message: {message}") # Avoid logging potentially large messages

    def _handle_ticker_message(self, message):
        """Process incoming ticker WebSocket messages and update funding rates"""
        try:
            # Basic validation
            if "topic" not in message or "data" not in message or not isinstance(message["data"], dict):
                # logger.debug(f"Ignoring non-ticker or invalid ticker message format: {message}")
                return

            topic = message["topic"]
            # Expected format: publicTrade.{symbol} or ticker.{symbol} (e.g., ticker.BTCUSDT)
            parts = topic.split('.')
            if len(parts) < 2:
                 logger.warning(f"Ignoring message with unexpected ticker topic format: {topic}")
                 return
            symbol = parts[1] 

            # Ignore if symbol not monitored
            if symbol not in self.symbols:
                 # logger.debug(f"Ignoring ticker message for unmonitored symbol: {symbol}")
                 return 

            data = message["data"]
            funding_rate_str = data.get("fundingRate")

            if funding_rate_str is not None and funding_rate_str != "":
                try:
                    new_funding_rate = float(funding_rate_str)
                    with self.lock:
                        # Update only if changed (optional, reduces lock contention slightly)
                        # if self.funding_rates.get(symbol) != new_funding_rate:
                        self.funding_rates[symbol] = new_funding_rate
                        # logger.debug(f"Updated funding rate for {symbol}: {new_funding_rate}")
                except ValueError:
                    logger.warning(f"Could not convert funding rate string '{funding_rate_str}' to float for {symbol}.")
            # else: 
                # logger.debug(f"No fundingRate field in ticker update for {symbol}")

        except Exception as e:
            # Safely get symbol for logging
            symbol_for_log = "unknown_symbol"
            try:
                topic = message.get("topic", "")
                parts = topic.split('.')
                if len(parts) >= 2:
                    symbol_for_log = parts[1]
            except Exception:
                pass 
            logger.error(f"Error processing ticker message for {symbol_for_log}: {e}", exc_info=True)
            # logger.error(f"Problematic ticker message: {message}")

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
                          # Use event wait for stoppable sleep
                          self._stop_event.wait(timeout=backoff)
                          if self._stop_event.is_set(): break # Exit if stopped during wait

                     self.init_websocket()
                     if not self.ws:
                         logger.error("Failed to initialize WebSocket. Retrying...")
                         self.reconnect_count += 1
                         continue # Retry after delay

                     # Fetch symbols if needed (usually only first time)
                     # Initialize price history also fetches initial funding rates now
                     if not self.symbols:
                          if not self.initialize_price_history():
                              logger.error("Failed initial price history load. Cannot proceed with subscription.")
                              # Consider a longer retry delay or stopping if this fails repeatedly
                              self.reconnect_count += 1
                              continue

                     # Subscribe to streams (kline and ticker)
                     if not self.subscribe_to_streams(): # Changed from subscribe_to_klines
                         logger.error("Failed to subscribe to streams. Retrying...")
                         self.reconnect_count += 1
                         self.connected = False # Ensure marked as disconnected
                         # Close the potentially partially connected socket before retry
                         if self.ws:
                            try:
                              self.ws.exit()
                            except Exception: pass
                            finally: self.ws = None
                         continue # Retry subscription after delay
                     else:
                          # Success! Reset reconnect count
                          self.reconnect_count = 0
                          self.connected = True
                          logger.info("Successfully connected and subscribed to WebSocket streams.")


                # If connected, just sleep and let the WS thread handle messages
                # The heartbeat monitor will trigger reconnect if needed
                # Check connection status before sleeping
                if not self.connected:
                    logger.info("Main Loop: Detected disconnected state. Initiating reconnection sequence.")
                    # No need to sleep, the loop will restart the connection attempt
                    continue
                    
                # Use stoppable sleep instead of time.sleep
                self._stop_event.wait(timeout=5) 

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
                 if self.ws:
                    try: self.ws.exit()
                    except Exception: pass
                    finally: self.ws = None


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

    with monitor.lock:  # Use lock to safely access shared data
        # --- Basic Monitor Status --- 
        rsi_count = len(monitor.rsi_values)
        funding_rate_count = len(monitor.funding_rates)
        tracked_symbols = len(monitor.symbols)
        subscribed_kline_count = len(monitor.subscribed_symbols)
        subscribed_ticker_count = len(monitor.subscribed_tickers)

        # Calculate time since last message
        time_since_last_msg = int(time.time() - monitor.last_message_time) if monitor.last_message_time else None

        # More accurate connection status determination
        if not monitor.connected:
            websocket_status = "disconnected"
        elif time_since_last_msg is None: # Should not happen if connected, but safety check
             websocket_status = "unknown"
        elif time_since_last_msg > 60: # Increased threshold for staleness
            websocket_status = "stale"  # Connection exists but no recent messages
        else:
            websocket_status = "connected"  # Actively receiving messages

        # --- RSI Ranking (using real-time RSI values) ---
        valid_rsi = {s: r for s, r in monitor.rsi_values.items() if pd.notna(r)}
        sorted_rsi = sorted(valid_rsi.items(), key=lambda item: item[1], reverse=True)
        top_5_rsi_symbols_data = sorted_rsi[:5]
        bottom_5_rsi_symbols_data = sorted_rsi[-5:]
        
        # Symbols needed for Daily RSI enrichment
        rsi_symbols_to_enrich = {s for s, r in top_5_rsi_symbols_data} | {s for s, r in bottom_5_rsi_symbols_data}

        # --- Funding Rate Ranking (using real-time funding rates from WebSocket) ---
        # Create a snapshot of the funding rates under lock
        current_funding_rates = monitor.funding_rates.copy()
        
        # Filter out symbols with invalid rates (e.g., None, although WS should provide floats)
        valid_funding_rates = [(s, r) for s, r in current_funding_rates.items() 
                               if isinstance(r, (int, float)) and s in monitor.symbols] # Ensure symbol is still monitored
        
        # Sort ascending for lowest, descending for highest
        sorted_funding_asc = sorted(valid_funding_rates, key=lambda item: item[1])
        sorted_funding_desc = sorted(valid_funding_rates, key=lambda item: item[1], reverse=True)

        # Get top/bottom 5
        top_5_funding_symbols_data = sorted_funding_desc[:5]
        bottom_5_funding_symbols_data = sorted_funding_asc[:5]
        
        # Format funding rate results
        results_top_5_funding = []
        for symbol, rate in top_5_funding_symbols_data:
             results_top_5_funding.append({
                 "symbol": symbol,
                 "funding_rate": rate,
                 "funding_rate_percent": f"{rate * 100:.4f}%"
             })

        results_bottom_5_funding = []
        for symbol, rate in bottom_5_funding_symbols_data:
             results_bottom_5_funding.append({
                 "symbol": symbol,
                 "funding_rate": rate,
                 "funding_rate_percent": f"{rate * 100:.4f}%"
             })

        # --- Enrichment: Fetch Daily RSI for Top/Bottom RSI Symbols ---
        # This part still requires HTTP calls as daily RSI is not on WebSocket
        daily_rsi_values = {}
        if rsi_symbols_to_enrich: # Only fetch if there are symbols to enrich
            logger.info(f"Fetching daily RSI for {len(rsi_symbols_to_enrich)} symbols...")
            symbols_processed_count = 0
            fetch_start_time = time.time()
            for symbol in rsi_symbols_to_enrich:
                # Check cache first
                cache_key = f"{symbol}_D"
                now = time.time()
                cached_val = monitor.daily_rsi_cache.get(cache_key)
                cache_time = monitor.daily_rsi_cache_time.get(cache_key, 0)
                
                if cached_val is not None and (now - cache_time < 3600): # Use 1-hour cache
                     daily_rsi = cached_val
                     logger.debug(f"Using cached daily RSI for {symbol}")
                else:
                    daily_rsi = monitor.get_rsi_for_interval(symbol, "D") # Use "D" for daily
                    # Update cache (even if None, to avoid re-fetching immediately)
                    monitor.daily_rsi_cache[cache_key] = daily_rsi
                    monitor.daily_rsi_cache_time[cache_key] = now
                
                # Store the result (could be None)
                daily_rsi_values[symbol] = round(daily_rsi, 2) if daily_rsi is not None else None

                symbols_processed_count += 1
                # Adaptive delay only if not using cache
                if daily_rsi is None or not (cached_val is not None and (now - cache_time < 3600)):
                     base_delay = 0.05
                     progress_factor = 1 - (symbols_processed_count / len(rsi_symbols_to_enrich))
                     delay = base_delay + (progress_factor * 0.1)
                     time.sleep(delay)
                     
            fetch_duration = time.time() - fetch_start_time
            logger.info(f"Finished fetching/caching daily RSI for {symbols_processed_count} symbols in {fetch_duration:.2f} seconds.")
        else:
             logger.info("No RSI symbols require daily RSI enrichment.")


        # --- Format Final RSI Results with Enrichment ---
        results_top_5_rsi = []
        for symbol, rsi in top_5_rsi_symbols_data:
            rate = current_funding_rates.get(symbol) # Get rate from WS data snapshot
            daily_rsi = daily_rsi_values.get(symbol)
            results_top_5_rsi.append({
                "symbol": symbol,
                "rsi": round(rsi, 2) if pd.notna(rsi) else None,
                "daily_rsi": daily_rsi,
                "funding_rate": rate,
                "funding_rate_percent": f"{rate * 100:.4f}%" if rate is not None else None
            })

        results_bottom_5_rsi = []
        for symbol, rsi in bottom_5_rsi_symbols_data:
            rate = current_funding_rates.get(symbol) # Get rate from WS data snapshot
            daily_rsi = daily_rsi_values.get(symbol)
            results_bottom_5_rsi.append({
                "symbol": symbol,
                "rsi": round(rsi, 2) if pd.notna(rsi) else None,
                "daily_rsi": daily_rsi,
                "funding_rate": rate,
                "funding_rate_percent": f"{rate * 100:.4f}%" if rate is not None else None
            })

    # Construct the final response dictionary
    response_data = {
        "status": "running" if monitor.is_running else "stopped",
        "websocket_status": websocket_status,
        "websocket_connected": monitor.connected,
        "last_message_time_ago_s": time_since_last_msg,
        "tracked_symbol_count": tracked_symbols,
        "subscribed_kline_count": subscribed_kline_count,
        "subscribed_ticker_count": subscribed_ticker_count,
        "rsi_calculated_count": rsi_count,
        "funding_rate_count": funding_rate_count,
        "reconnect_attempts": monitor.reconnect_count,
        "top_5_rsi": results_top_5_rsi,
        "bottom_5_rsi": results_bottom_5_rsi,
        "top_5_funding_rate": results_top_5_funding,
        "bottom_5_funding_rate": results_bottom_5_funding
    }
    
    return response_data

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