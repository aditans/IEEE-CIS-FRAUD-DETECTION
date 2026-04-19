# Live Multi-Device Fraud Graph Demo

This guide gives you a clean presentation flow for tomorrow:
- phones/laptops send transactions,
- a live graph updates in real-time,
- suspicious fraud components appear in a side panel.

## 1) Install dependencies

```powershell
pip install -r requirements.txt
```

## 2) Start the live demo server

```powershell
python phase4_live_demo.py --host 0.0.0.0 --port 5000
```

Open locally:
- Dashboard: http://localhost:5000/
- Sender page: http://localhost:5000/submit

## 3) Share to other devices with ngrok

In a second terminal:

```powershell
ngrok http 5000
```

Copy your ngrok public URL, for example:
- https://abc123.ngrok-free.app

Then share:
- Dashboard URL: https://abc123.ngrok-free.app/
- Sender URL: https://abc123.ngrok-free.app/submit

Your friends can open `/submit` and send transactions from their phones.
The main dashboard updates every 2 seconds.

## 4) Demo script for PPT

1. Open dashboard on projector.
2. Click `Seed 20 Tx` once to show initial graph structure.
3. Ask 2-3 friends to open `/submit` and send transactions.
4. Send a few high-amount or repeated transactions from same source.
5. Point to:
   - red/orange high-risk nodes,
   - fraud transaction count and fraud rate,
   - `Fraud Components` panel showing suspicious connected subgraphs.

## 5) Suggested transaction patterns (to trigger fraud)

Use these values on sender page:
- Source: `acct_1001`, Amount: `9000`, Device: `dev_99`, send 2-3 times quickly.
- Source: `acct_1001`, Amount: `7000`, Device: `dev_100`, send quickly again.
- Source: `acct_1012`, Amount: `6000`, Device: `dev_4`.

This usually triggers velocity + amount + new-device signals, making fraud scores jump.

## Notes

- This app is a live explainable demo layer, not your final offline training pipeline.
- It is designed for interpretability in presentations: "which network pockets are suspicious right now".
