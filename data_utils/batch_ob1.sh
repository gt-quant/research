for symbol in BTCUSDT ETHUSDT; do
  for date in 20250111 20250112 20250113; do
    sh data_utils/pull_ob1_from_server.sh "$symbol" "$date"
  done
done
