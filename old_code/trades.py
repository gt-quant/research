from pybit.unified_trading import WebSocket
from time import sleep

# Create an instance of the WebSocket
ws = WebSocket(testnet=False, channel_type="linear")

# Callback function to handle the incoming trade messages
def handle_message(message):
    # Check if the message contains trade data
    if 'data' in message:
        print(message['data'])
        # for trade in message['data']:
        #     # Extract and print relevant details from the trade message
        #     print(f"Timestamp: {trade['T']}")
        #     print(f"Symbol: {trade['s']}")
        #     print(f"Side: {trade['S']}")
        #     print(f"Trade Size: {trade['v']}")
        #     print(f"Trade Price: {trade['p']}")
        #     print(f"Price Change Direction: {trade['L']}")
        #     print(f"Trade ID: {trade['i']}")
        #     print(f"Is Block Trade: {trade['BT']}")
        #     print(f"Is RPI Trade: {trade['RPI']}")
        #     print(f"Mark Price: {trade.get('mP', 'N/A')}")
        #     print(f"Index Price: {trade.get('iP', 'N/A')}")
        #     print(f"Mark IV: {trade.get('mIv', 'N/A')}")
        #     print(f"IV: {trade.get('iv', 'N/A')}")
        #     print("-" * 50)  # Separator for better readability

# Subscribe to the trade stream for the BTCUSDT symbol
ws.trade_stream(symbol="LOOKSUSDT", callback=handle_message)

# Keep the script running to continuously receive trade messages
while True:
    sleep(1)
