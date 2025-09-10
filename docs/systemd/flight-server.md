# Flight Server systemd setup

Copy the provided service file and enable it so the Arrow Flight server
starts at boot:

```bash
sudo cp docs/systemd/flight-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now flight-server.service
```

The service runs `botcopier-flight-server` from `/opt/BotCopier` and
persists incoming trade and metric batches under `/opt/BotCopier/flight_logs`.
Logs are mirrored to the systemd journal and can be viewed with:

```bash
sudo journalctl -u flight-server.service -f
```
