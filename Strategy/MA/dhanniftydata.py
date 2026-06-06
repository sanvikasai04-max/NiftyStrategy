



from dhanhq import dhanhq, DhanContext
import pandas as pd
import json


CLIENT_ID = "1107485546"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzgwODU4MTg5LCJpYXQiOjE3ODA3NzE3ODksInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA3NDg1NTQ2In0.BuUgZkqjMuZIGMOrEfdhYWMIsJiGH__0-IS0jTA0YqJCR41tNCkkAURQSSJJOmr_izP9AgLQ7PjPHGEFsiLrYQ"


dhan_context = DhanContext(CLIENT_ID, ACCESS_TOKEN)
client = dhanhq(dhan_context)

data = client.intraday_minute_data(
    security_id="13",
    exchange_segment="IDX_I",
    instrument_type="INDEX",
    interval=1,
    from_date="2026-03-4",
    to_date="2026-06-01",
)

#print(json.dumps(data, indent=2))


if data.get("status") == "success":

    df = pd.DataFrame({
        "timestamp": data["data"]["timestamp"],
        "open": data["data"]["open"],
        "high": data["data"]["high"],
        "low": data["data"]["low"],
        "close": data["data"]["close"],
        "volume": data["data"]["volume"]
    })

    df["datetime"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.strftime("%d-%m-%Y %H:%M")
    )

    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    print(df.head())

    df.to_csv("nifty_1min_clean.csv", index=False)

    #print("Saved nifty_5min_clean.csv")

else:
    print("API failed:", data)