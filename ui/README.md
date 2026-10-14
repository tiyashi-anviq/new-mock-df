# UI (React + Vite + Carbon)

Screens for the Eveready demand forecasting demo. It talks to the API in `..\api` and never reads the data files itself.

## Run (two PowerShell windows)

Window 1, the API (must be on port 4004):

```
cd ..\api
python app.py
```

Window 2, the UI, from this folder:

```
npm install        # first time only
npm run dev
```

Open http://localhost:3004 . The top right of every page shows "API connected" once the two can talk.

## Built step by step

1. Skeleton, left rail, page header, live API status (this step)
2. Forecast & accuracy: M1, M2, M3 views
3. Challenge flags, Why this week, Overview
